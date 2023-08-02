/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CudaMgr/CudaMgr.h"
#include "GpuSharedMemoryTestHelpers.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Compiler/CodegenTraitsDescriptor.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "QueryEngine/NvidiaKernel.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "Shared/TargetInfo.h"
#include "TestHelpers.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

extern bool g_is_test_env;

auto int8_type = hdk::ir::Context::defaultCtx().int8();
auto int16_type = hdk::ir::Context::defaultCtx().int16();
auto int32_type = hdk::ir::Context::defaultCtx().int32();
auto int64_type = hdk::ir::Context::defaultCtx().int64();
auto float_type = hdk::ir::Context::defaultCtx().fp32();
auto double_type = hdk::ir::Context::defaultCtx().fp64();

class CudaReductionTester : public GpuReductionTester {
 public:
  CudaReductionTester(const Config& config,
                      llvm::Module* module,
                      llvm::LLVMContext& context,
                      const QueryMemoryDescriptor& qmd,
                      const std::vector<TargetInfo>& targets,
                      const std::vector<int64_t>& init_agg_values,
                      GpuMgr* gpu_mgr,
                      const compiler::CodegenTraits& traits,
                      Executor* executor)
      : GpuReductionTester(config,
                           module,
                           context,
                           qmd,
                           targets,
                           init_agg_values,
                           gpu_mgr,
                           traits,
                           executor) {}
  void performReductionTest(const std::vector<std::unique_ptr<ResultSet>>& result_sets,
                            const ResultSetStorage* gpu_result_storage,
                            const size_t device_id);
};

namespace {
void prepare_generated_cuda_kernel(llvm::Module* module,
                                   llvm::LLVMContext& context,
                                   llvm::Function* kernel) {
  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(kernel),
                               llvm::MDString::get(context, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(context), 1))};

  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(context, md_vals));
}

std::unique_ptr<CudaDeviceCompilationContext> compile_and_link_cuda_code(
    const std::string& cuda_llir,
    llvm::Module* module,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const std::string& kernel_name,
    const size_t gpu_block_size = 1024,
    const size_t gpu_device_idx = 0) {
  CHECK(module);
  CHECK(cuda_mgr);
  auto& context = module->getContext();
  std::unique_ptr<llvm::TargetMachine> nvptx_target_machine =
      compiler::CUDABackend::initializeNVPTXBackend(cuda_mgr->getDeviceArch());
  const auto ptx =
      compiler::CUDABackend::generatePTX(cuda_llir, nvptx_target_machine.get(), context);

  auto cubin_result = ptx_to_cubin(ptx, gpu_block_size, cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();
  auto gpu_context = std::make_unique<CudaDeviceCompilationContext>(cubin,
                                                                    kernel_name,
                                                                    gpu_device_idx,
                                                                    cuda_mgr,
                                                                    num_options,
                                                                    &option_keys[0],
                                                                    &option_values[0]);

  checkCudaErrors(cuLinkDestroy(link_state));
  return gpu_context;
}

void perform_test_and_verify_results(TestInputData input) {
  const auto platform = GpuMgrPlatform::CUDA;
  const bool is_l0 = platform == GpuMgrPlatform::L0;
  auto executor = Executor::getExecutor(nullptr, nullptr);
  auto& context = executor->getContext();
  auto cgen_state = std::unique_ptr<CgenState>(
      new CgenState({}, false, false, executor->getExtensionModuleContext(), context));
  cgen_state->set_module_shallow_copy(
      executor->getExtensionModuleContext()->getRTModule(is_l0));
  auto module = cgen_state->module_;
  auto cgen_traits = get_codegen_traits(platform);
  module->setDataLayout(cgen_traits.dataLayout());
  module->setTargetTriple(cgen_traits.triple());
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
  const auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(nullptr, Executor::getArenaBlockSize());
  auto query_mem_desc = perfect_hash_one_col_desc(
      input.target_infos, input.suggested_agg_widths, input.min_entry, input.max_entry);
  if (input.keyless_hash) {
    query_mem_desc.setHasKeylessHash(true);
    query_mem_desc.setTargetIdxForKey(input.target_index_for_key);
  }

  std::vector<StrideNumberGenerator> generators(
      input.num_input_buffers, StrideNumberGenerator(1, input.step_size));
  std::vector<size_t> steps(input.num_input_buffers, input.step_size);
  auto input_result_sets = create_and_fill_input_result_sets(input.num_input_buffers,
                                                             row_set_mem_owner,
                                                             query_mem_desc,
                                                             input.target_infos,
                                                             generators,
                                                             steps);

  const auto [cpu_result_set, gpu_result_set] = create_and_init_output_result_sets(
      row_set_mem_owner, query_mem_desc, input.target_infos);

  // performing reduciton using the GPU reduction code:
  Config config;
  CudaReductionTester gpu_smem_tester(
      config,
      module,
      context,
      query_mem_desc,
      input.target_infos,
      init_agg_val_vec(input.target_infos, query_mem_desc),
      cuda_mgr.get(),
      cgen_traits,
      executor.get());
  gpu_smem_tester.codegen(CompilationOptions::defaults(
      ExecutorDeviceType::GPU,
      is_l0));  // generate code for gpu reduciton and initialization
  gpu_smem_tester.codegenWrapperKernel();
  gpu_smem_tester.performReductionTest(
      input_result_sets, gpu_result_set->getStorage(), input.device_id);

  // CPU reduction for validation:
  perform_reduction_on_cpu(input_result_sets, cpu_result_set->getStorage());

  const auto cmp_result =
      std::memcmp(cpu_result_set->getStorage()->getUnderlyingBuffer(),
                  gpu_result_set->getStorage()->getUnderlyingBuffer(),
                  query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU));
  ASSERT_EQ(cmp_result, 0);
}

}  // namespace

void CudaReductionTester::performReductionTest(
    const std::vector<std::unique_ptr<ResultSet>>& result_sets,
    const ResultSetStorage* gpu_result_storage,
    const size_t device_id) {
  prepare_generated_cuda_kernel(module_, context_, getWrapperKernel());

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  module_->print(os, nullptr);
  os.flush();
  std::string module_str(ss.str());

  std::unique_ptr<CudaDeviceCompilationContext> gpu_context(
      compile_and_link_cuda_code(module_str,
                                 module_,
                                 dynamic_cast<CudaMgr_Namespace::CudaMgr*>(gpu_mgr_),
                                 getWrapperKernel()->getName().str()));

  const auto buffer_size = query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU);
  const size_t num_buffers = result_sets.size();
  std::vector<int8_t*> d_input_buffers;
  for (size_t i = 0; i < num_buffers; i++) {
    d_input_buffers.push_back(gpu_mgr_->allocateDeviceMem(buffer_size, device_id));
    gpu_mgr_->copyHostToDevice(d_input_buffers[i],
                               result_sets[i]->getStorage()->getUnderlyingBuffer(),
                               buffer_size,
                               device_id);
  }

  constexpr size_t num_kernel_params = 3;
  CHECK_EQ(getWrapperKernel()->arg_size(), num_kernel_params);

  // parameter 1: an array of device pointers
  std::vector<CUdeviceptr> h_input_buffer_dptrs;
  h_input_buffer_dptrs.reserve(num_buffers);
  std::transform(d_input_buffers.begin(),
                 d_input_buffers.end(),
                 std::back_inserter(h_input_buffer_dptrs),
                 [](int8_t* dptr) { return reinterpret_cast<CUdeviceptr>(dptr); });

  auto d_input_buffer_dptrs =
      gpu_mgr_->allocateDeviceMem(num_buffers * sizeof(CUdeviceptr), device_id);
  gpu_mgr_->copyHostToDevice(d_input_buffer_dptrs,
                             reinterpret_cast<int8_t*>(h_input_buffer_dptrs.data()),
                             num_buffers * sizeof(CUdeviceptr),
                             device_id);

  // parameter 2: number of buffers
  auto d_num_buffers = gpu_mgr_->allocateDeviceMem(sizeof(int64_t), device_id);
  gpu_mgr_->copyHostToDevice(d_num_buffers,
                             reinterpret_cast<const int8_t*>(&num_buffers),
                             sizeof(int64_t),
                             device_id);

  // parameter 3: device pointer to the output buffer
  auto d_result_buffer = gpu_mgr_->allocateDeviceMem(buffer_size, device_id);
  gpu_mgr_->copyHostToDevice(
      d_result_buffer, gpu_result_storage->getUnderlyingBuffer(), buffer_size, device_id);

  // collecting all kernel parameters:
  std::vector<CUdeviceptr> h_kernel_params{
      reinterpret_cast<CUdeviceptr>(d_input_buffer_dptrs),
      reinterpret_cast<CUdeviceptr>(d_num_buffers),
      reinterpret_cast<CUdeviceptr>(d_result_buffer)};

  // casting each kernel parameter to be a void* device ptr itself:
  std::vector<void*> kernel_param_ptrs;
  kernel_param_ptrs.reserve(num_kernel_params);
  std::transform(h_kernel_params.begin(),
                 h_kernel_params.end(),
                 std::back_inserter(kernel_param_ptrs),
                 [](CUdeviceptr& param) { return &param; });

  // launching a kernel:
  auto cu_func = static_cast<CUfunction>(gpu_context->kernel());
  // we launch as many threadblocks as there are input buffers:
  // in other words, each input buffer is handled by a single threadblock.

  checkCudaErrors(cuLaunchKernel(cu_func,
                                 num_buffers,
                                 1,
                                 1,
                                 1024,
                                 1,
                                 1,
                                 buffer_size,
                                 0,
                                 kernel_param_ptrs.data(),
                                 nullptr));

  // transfer back the results:
  gpu_mgr_->copyDeviceToHost(
      gpu_result_storage->getUnderlyingBuffer(), d_result_buffer, buffer_size, device_id);

  // release the gpu memory used:
  for (auto& d_buffer : d_input_buffers) {
    gpu_mgr_->freeDeviceMem(d_buffer);
  }
  gpu_mgr_->freeDeviceMem(d_input_buffer_dptrs);
  gpu_mgr_->freeDeviceMem(d_num_buffers);
  gpu_mgr_->freeDeviceMem(d_result_buffer);
}

TEST(SingleColumn, VariableEntries_CountQuery_4B_Group) {
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241, 511, 1021}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {4}, {hdk::ir::AggType::kCount}, {int32_type}, {int32_type}))
        .setAggWidth(4)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableEntries_CountQuery_8B_Group) {
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241, 511, 1021}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {8}, {hdk::ir::AggType::kCount}, {int64_type}, {int64_type}))
        .setAggWidth(8)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_1) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kMin,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kAvg},
          {int64_type, int64_type, int64_type, int64_type, double_type},
          {int32_type, int32_type, int32_type, int32_type, int32_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_2) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMin},
          {int64_type, double_type, int64_type, int64_type, int64_type},
          {int32_type, int32_type, int32_type, int32_type, int32_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_3) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(367)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMin},
          {int64_type, double_type, double_type, double_type, double_type},
          {int32_type, double_type, double_type, double_type, double_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_4) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(517)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMin},
          {int64_type, float_type, float_type, float_type, float_type},
          {int16_type, float_type, float_type, float_type, float_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableNumBuffers) {
  TestInputData input;
  input.setDeviceId(0)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(266)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kMin},
          {int32_type, int64_type, double_type, float_type, double_type},
          {int8_type, int8_type, int16_type, float_type, double_type}));

  for (auto& num_buffers : {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128}) {
    input.setNumInputBuffers(num_buffers);
    perform_test_and_verify_results(input);
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
