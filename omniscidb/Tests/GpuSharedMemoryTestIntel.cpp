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

#include "GpuSharedMemoryTestIntel.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Compiler/CodegenTraitsDescriptor.h"
#include "QueryEngine/Compiler/HelperFunctions.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/ResultSetReduction.h"
#include "QueryEngine/ResultSetReductionJIT.h"

extern bool g_is_test_env;

auto int8_type = hdk::ir::Context::defaultCtx().int8();
auto int16_type = hdk::ir::Context::defaultCtx().int16();
auto int32_type = hdk::ir::Context::defaultCtx().int32();
auto int64_type = hdk::ir::Context::defaultCtx().int64();
auto float_type = hdk::ir::Context::defaultCtx().fp32();
auto double_type = hdk::ir::Context::defaultCtx().fp64();

namespace {

compiler::CodegenTraits get_codegen_traits() {
  return compiler::CodegenTraits::get(compiler::cuda_cgen_traits_desc);
}

void init_storage_buffer(int8_t* buffer,
                         const std::vector<TargetInfo>& targets,
                         const QueryMemoryDescriptor& query_mem_desc) {
  // get the initial values for all the aggregate columns
  const auto init_agg_vals = init_agg_val_vec(targets, query_mem_desc);
  CHECK(!query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);

  const auto row_size = query_mem_desc.getRowSize();
  CHECK(query_mem_desc.hasKeylessHash());
  for (size_t entry_idx = 0; entry_idx < query_mem_desc.getEntryCount(); ++entry_idx) {
    const auto row_ptr = buffer + entry_idx * row_size;
    size_t init_agg_idx{0};
    int64_t init_val{0};
    // initialize each row's aggregate columns:
    auto col_ptr = row_ptr + query_mem_desc.getColOffInBytes(0);
    for (size_t slot_idx = 0; slot_idx < query_mem_desc.getSlotCount(); slot_idx++) {
      if (query_mem_desc.getPaddedSlotWidthBytes(slot_idx) > 0) {
        init_val = init_agg_vals[init_agg_idx++];
      }
      switch (query_mem_desc.getPaddedSlotWidthBytes(slot_idx)) {
        case 4:
          *reinterpret_cast<int32_t*>(col_ptr) = static_cast<int32_t>(init_val);
          break;
        case 8:
          *reinterpret_cast<int64_t*>(col_ptr) = init_val;
          break;
        case 0:
          break;
        default:
          UNREACHABLE();
      }
      col_ptr += query_mem_desc.getNextColOffInBytes(col_ptr, entry_idx, slot_idx);
    }
  }
}

}  // namespace

void GpuReductionTester::codegenWrapperKernel() {
  const unsigned address_space = 0;
  auto pi8_type = llvm::Type::getInt8PtrTy(context_, address_space);
  std::vector<llvm::Type*> input_arguments;
  input_arguments.push_back(llvm::PointerType::get(pi8_type, address_space));
  input_arguments.push_back(llvm::Type::getInt64Ty(context_));  // num input buffers
  input_arguments.push_back(llvm::Type::getInt8PtrTy(context_, address_space));

  llvm::FunctionType* ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context_), input_arguments, false);
  wrapper_kernel_ = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "wrapper_kernel", module_);

  auto arg_it = wrapper_kernel_->arg_begin();
  auto input_ptrs = &*arg_it;
  input_ptrs->setName("input_pointers");
  arg_it++;
  auto num_buffers = &*arg_it;
  num_buffers->setName("num_buffers");
  arg_it++;
  auto output_buffer = &*arg_it;
  output_buffer->setName("output_buffer");

  llvm::IRBuilder<> ir_builder(context_);

  auto bb_entry = llvm::BasicBlock::Create(context_, ".entry", wrapper_kernel_);
  auto bb_body = llvm::BasicBlock::Create(context_, ".body", wrapper_kernel_);
  auto bb_exit = llvm::BasicBlock::Create(context_, ".exit", wrapper_kernel_);

  // return if blockIdx.x > num_buffers
  ir_builder.SetInsertPoint(bb_entry);
  auto get_block_index_func = getFunction("get_block_index");
  auto block_index = ir_builder.CreateCall(get_block_index_func, {}, "block_index");
  const auto is_block_inbound =
      ir_builder.CreateICmpSLT(block_index, num_buffers, "is_block_inbound");
  ir_builder.CreateCondBr(is_block_inbound, bb_body, bb_exit);

  // locate the corresponding input buffer:
  ir_builder.SetInsertPoint(bb_body);
  auto input_buffer_gep = ir_builder.CreateGEP(
      input_ptrs->getType()->getScalarType()->getPointerElementType(),
      input_ptrs,
      block_index);
  auto input_buffer = ir_builder.CreateLoad(
      llvm::Type::getInt8PtrTy(context_, address_space), input_buffer_gep);
  auto input_buffer_ptr = ir_builder.CreatePointerCast(
      input_buffer, llvm::Type::getInt64PtrTy(context_, 4), "input_buffer_ptr");
  const auto buffer_size = ll_int(
      static_cast<int32_t>(query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)),
      context_);

  // initializing shared memory and copy input buffer into shared memory buffer:
  auto init_smem_func = getFunction("init_shared_mem");
  auto smem_input_buffer_ptr = ir_builder.CreateCall(init_smem_func,
                                                     {
                                                         input_buffer_ptr,
                                                         buffer_size,
                                                     },
                                                     "smem_input_buffer_ptr");

  auto smem_input_buffer_ptr1 =
      ir_builder.CreatePointerCast(smem_input_buffer_ptr,
                                   llvm::Type::getInt64PtrTy(context_, 4),
                                   "smem_input_buffer_ptr");

  auto output_buffer_ptr = ir_builder.CreatePointerCast(
      output_buffer, llvm::Type::getInt64PtrTy(context_, 4), "output_buffer_ptr");
  // call the reduction function
  CHECK(reduction_func_);
  std::vector<llvm::Value*> reduction_args{
      output_buffer_ptr, smem_input_buffer_ptr1, buffer_size};
  ir_builder.CreateCall(reduction_func_, reduction_args);
  ir_builder.CreateBr(bb_exit);

  ir_builder.SetInsertPoint(bb_exit);
  ir_builder.CreateRet(nullptr);

  wrapper_kernel_->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
}

namespace {
void prepare_generated_gpu_kernel(llvm::Module* module,
                                  llvm::LLVMContext& context,
                                  llvm::Function* kernel) {
  // might be extra, remove and clean up
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("spir64-unknown-unknown");

  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(kernel),
                               llvm::MDString::get(context, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(context), 1))};

  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(context, md_vals));
}

std::unique_ptr<L0DeviceCompilationContext> compile_and_link_gpu_code(
    const std::string& l0_llir,
    llvm::Module* module,
    l0::L0Manager* l0_mgr,
    const std::string& kernel_name,
    const size_t gpu_block_size = 1024,
    const size_t gpu_device_idx = 0) {
  CHECK(module);
  CHECK(l0_mgr);

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::cout << "compile_and_link_gpu_code - before writeSpirv" << std::endl;

  std::ostringstream ss;
  std::string err;
  auto success = writeSpirv(module, opts, ss, err);
  CHECK(success) << "Spirv translation failed with error: " << err << "\n";

  std::cout << "compile_and_link_gpu_code - before spv_to_bin" << std::endl;

  L0BinResult bin_result;
  bin_result = spv_to_bin(ss.str(), kernel_name, gpu_block_size, l0_mgr);

  std::cout << "compile_and_link_gpu_code - after spv_to_bin" << std::endl;

  auto l0_context = std::make_unique<L0DeviceCompilationContext>(
      bin_result.device, bin_result.kernel, bin_result.module, l0_mgr, 0, 1);

  return l0_context;
}

std::vector<std::unique_ptr<ResultSet>> create_and_fill_input_result_sets(
    const size_t num_input_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& target_infos,
    std::vector<StrideNumberGenerator>& generators,
    const std::vector<size_t>& steps) {
  std::vector<std::unique_ptr<ResultSet>> result_sets;
  for (size_t i = 0; i < num_input_buffers; i++) {
    result_sets.push_back(std::make_unique<ResultSet>(target_infos,
                                                      ExecutorDeviceType::CPU,
                                                      query_mem_desc,
                                                      row_set_mem_owner,
                                                      nullptr,
                                                      0,
                                                      0));
    const auto storage = result_sets.back()->allocateStorage();
    fill_storage_buffer(storage->getUnderlyingBuffer(),
                        target_infos,
                        query_mem_desc,
                        generators[i],
                        steps[i]);
  }
  return result_sets;
}

std::pair<std::unique_ptr<ResultSet>, std::unique_ptr<ResultSet>>
create_and_init_output_result_sets(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const std::vector<TargetInfo>& target_infos) {
  // CPU result set, will eventually host CPU reduciton results for validations
  auto cpu_result_set = std::make_unique<ResultSet>(target_infos,
                                                    ExecutorDeviceType::CPU,
                                                    query_mem_desc,
                                                    row_set_mem_owner,
                                                    nullptr,
                                                    0,
                                                    0);
  auto cpu_storage_result = cpu_result_set->allocateStorage();
  init_storage_buffer(
      cpu_storage_result->getUnderlyingBuffer(), target_infos, query_mem_desc);

  // GPU result set, will eventually host GPU reduction results
  auto gpu_result_set = std::make_unique<ResultSet>(target_infos,
                                                    ExecutorDeviceType::GPU,
                                                    query_mem_desc,
                                                    row_set_mem_owner,
                                                    nullptr,
                                                    0,
                                                    0);
  auto gpu_storage_result = gpu_result_set->allocateStorage();
  init_storage_buffer(
      gpu_storage_result->getUnderlyingBuffer(), target_infos, query_mem_desc);
  return std::make_pair(std::move(cpu_result_set), std::move(gpu_result_set));
}
void perform_reduction_on_cpu(std::vector<std::unique_ptr<ResultSet>>& result_sets,
                              const ResultSetStorage* cpu_result_storage) {
  CHECK(result_sets.size() > 0);
  Config config;
  // for codegen only
  auto executor = Executor::getExecutor(nullptr);
  ResultSetReductionJIT reduction_jit(result_sets.front()->getQueryMemDesc(),
                                      result_sets.front()->getTargetInfos(),
                                      result_sets.front()->getTargetInitVals(),
                                      config,
                                      executor.get());
  const auto reduction_code = reduction_jit.codegen();
  for (auto& result_set : result_sets) {
    ResultSetReduction::reduce(*cpu_result_storage,
                               *(result_set->getStorage()),
                               {},
                               reduction_code,
                               config,
                               executor.get());
  }
}

struct TestInputData {
  size_t device_id;
  size_t num_input_buffers;
  std::vector<TargetInfo> target_infos;
  int8_t suggested_agg_widths;
  size_t min_entry;
  size_t max_entry;
  size_t step_size;
  bool keyless_hash;
  int32_t target_index_for_key;
  TestInputData()
      : device_id(0)
      , num_input_buffers(0)
      , suggested_agg_widths(0)
      , min_entry(0)
      , max_entry(0)
      , step_size(2)
      , keyless_hash(false)
      , target_index_for_key(0) {}
  TestInputData& setDeviceId(const size_t id) {
    device_id = id;
    return *this;
  }
  TestInputData& setNumInputBuffers(size_t num_buffers) {
    num_input_buffers = num_buffers;
    return *this;
  }
  TestInputData& setTargetInfos(std::vector<TargetInfo> tis) {
    target_infos = tis;
    return *this;
  }
  TestInputData& setAggWidth(int8_t agg_width) {
    suggested_agg_widths = agg_width;
    return *this;
  }
  TestInputData& setMinEntry(size_t min_e) {
    min_entry = min_e;
    return *this;
  }
  TestInputData& setMaxEntry(size_t max_e) {
    max_entry = max_e;
    return *this;
  }
  TestInputData& setKeylessHash(bool is_keyless) {
    keyless_hash = is_keyless;
    return *this;
  }
  TestInputData& setTargetIndexForKey(size_t target_idx) {
    target_index_for_key = target_idx;
    return *this;
  }
  TestInputData& setStepSize(size_t step) {
    step_size = step;
    return *this;
  }
};

void perform_test_and_verify_results(TestInputData input) {
  auto executor = Executor::getExecutor(nullptr, nullptr);
  auto& context = executor->getContext();
  auto cgen_state = std::unique_ptr<CgenState>(
      new CgenState({}, false, false, executor->getExtensionModuleContext(), context));
  cgen_state->set_module_shallow_copy(
      executor->getExtensionModuleContext()->getRTModule(/*is_l0=*/true));
  auto module = cgen_state->module_;
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("spir64-unknown-unknown");
  auto l0_mgr = std::make_unique<l0::L0Manager>();
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
  GpuReductionTester gpu_smem_tester(config,
                                     module,
                                     context,
                                     query_mem_desc,
                                     input.target_infos,
                                     init_agg_val_vec(input.target_infos, query_mem_desc),
                                     l0_mgr.get(),
                                     get_codegen_traits(),
                                     executor.get());
  gpu_smem_tester.codegen(CompilationOptions::defaults(
      ExecutorDeviceType::GPU,
      true));  // generate code for gpu reduciton and initialization
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

void insert_globals(llvm::Module* from, llvm::Module* to) {
  for (const llvm::GlobalVariable& I : from->globals()) {
    std::cerr << "Adding global " << I.getName().str() << std::endl;
    llvm::GlobalVariable* new_gv =
        new llvm::GlobalVariable(*to,
                                 I.getValueType(),
                                 I.isConstant(),
                                 I.getLinkage(),
                                 (llvm::Constant*)nullptr,
                                 I.getName(),
                                 (llvm::GlobalVariable*)nullptr,
                                 I.getThreadLocalMode(),
                                 I.getType()->getAddressSpace());
    new_gv->copyAttributesFrom(&I);
  }
}
void insert_declaration_tmp(llvm::Module* from,
                            llvm::Module* to,
                            const std::string& fname) {
  auto fn = from->getFunction(fname);
  CHECK(fn);

  llvm::Function::Create(
      fn->getFunctionType(), llvm::GlobalValue::ExternalLinkage, fn->getName(), *to);
}

void GpuReductionTester::performReductionTest(
    const std::vector<std::unique_ptr<ResultSet>>& result_sets,
    const ResultSetStorage* gpu_result_storage,
    const size_t device_id) {
  DUMP_MODULE(module_, "gen.ll");
  prepare_generated_gpu_kernel(module_, context_, getWrapperKernel());

  auto& ext_module = executor_->getExtensionModuleContext()->getSpirvHelperFuncModule();

  DUMP_MODULE(module_, "after.linking.before.insert_declaration.spirv.ll")

  for (auto& F : *ext_module) {
    insert_declaration_tmp(ext_module.get(), module_, F.getName().str());
  }

  insert_globals(ext_module.get(), module_);
  DUMP_MODULE(module_, "after.insert_global.spirv.ll")

  // Initialize shared memory buffer
  const auto slm_buffer = module_->getNamedGlobal("slm.buf.i64");
  CHECK(slm_buffer);
  llvm::ArrayType* ArrayTy_0 =
      llvm::ArrayType::get(llvm::IntegerType::get(module_->getContext(), 64), 1024);
  llvm::ConstantAggregateZero* const_array_2 =
      llvm::ConstantAggregateZero::get(ArrayTy_0);
  slm_buffer->setInitializer(const_array_2);

#ifdef DEBUG
  // Check global string name
  for (llvm::GlobalVariable& G : module_->getGlobalList()) {
    // std::ostringstream oss;
    std::cerr << " global var =  " << G.getName().str() << std::endl;
  }
#endif

  DUMP_MODULE(module_, "after.linking.before.replace_function.spirv.ll")

  for (auto& F : *ext_module) {
    if (!F.isDeclaration()) {
      compiler::replace_function(ext_module.get(), module_, F.getName().str());
    }
  }

  DUMP_MODULE(module_, "after.linking.spirv.ll")
  std::cout << "PerformReductionTest - after linking" << std::endl;

  // set proper calling conv & mangle spirv built-ins
  for (auto& Fn : *module_) {
    Fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
    if (Fn.getName().startswith("__spirv_")) {
      CHECK(Fn.isDeclaration());
      Fn.setName(compiler::mangle_spirv_builtin(Fn));
    }
  }

  std::cout << "PerformReductionTest - before calling conv" << std::endl;

  for (auto& Fn : *module_) {
    for (auto I = llvm::inst_begin(Fn), E = llvm::inst_end(Fn); I != E; ++I) {
      if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
        CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
      }
    }
  }
  std::cout << "PerformReductionTest - after calling conv" << std::endl;

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  module_->print(os, nullptr);
  os.flush();
  std::string module_str(ss.str());

  std::cout << "PerformReductionTest - before linking" << std::endl;

  std::unique_ptr<L0DeviceCompilationContext> gpu_context(compile_and_link_gpu_code(
      module_str, module_, l0_mgr_, getWrapperKernel()->getName().str()));
  std::cout << "PerformReductionTest - after compile_and_link_gpu_code" << std::endl;

  const auto buffer_size = query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU);
  const size_t num_buffers = result_sets.size();
  std::vector<int8_t*> d_input_buffers;
  for (size_t i = 0; i < num_buffers; i++) {
    d_input_buffers.push_back(l0_mgr_->allocateDeviceMem(buffer_size, device_id));
    l0_mgr_->copyHostToDevice(d_input_buffers[i],
                              result_sets[i]->getStorage()->getUnderlyingBuffer(),
                              buffer_size,
                              device_id);
  }

  constexpr size_t num_kernel_params = 3;
  CHECK_EQ(getWrapperKernel()->arg_size(), num_kernel_params);

  // parameter 1: an array of device pointers
  typedef int8_t* L0deviceptr;
  std::vector<L0deviceptr> h_input_buffer_dptrs;
  h_input_buffer_dptrs.reserve(num_buffers);
  std::transform(d_input_buffers.begin(),
                 d_input_buffers.end(),
                 std::back_inserter(h_input_buffer_dptrs),
                 [](int8_t* dptr) { return reinterpret_cast<L0deviceptr>(dptr); });

  auto d_input_buffer_dptrs =
      l0_mgr_->allocateDeviceMem(num_buffers * sizeof(L0deviceptr), device_id);
  l0_mgr_->copyHostToDevice(d_input_buffer_dptrs,
                            reinterpret_cast<int8_t*>(h_input_buffer_dptrs.data()),
                            num_buffers * sizeof(L0deviceptr),
                            device_id);

  // parameter 2: number of buffers
  auto d_num_buffers = l0_mgr_->allocateDeviceMem(sizeof(int64_t), device_id);
  l0_mgr_->copyHostToDevice(d_num_buffers,
                            reinterpret_cast<const int8_t*>(&num_buffers),
                            sizeof(int64_t),
                            device_id);

  // parameter 3: device pointer to the output buffer
  auto d_result_buffer = l0_mgr_->allocateDeviceMem(buffer_size, device_id);
  l0_mgr_->copyHostToDevice(
      d_result_buffer, gpu_result_storage->getUnderlyingBuffer(), buffer_size, device_id);

  // collecting all kernel parameters:
  std::vector<L0deviceptr> h_kernel_params{
      reinterpret_cast<L0deviceptr>(d_input_buffer_dptrs),
      reinterpret_cast<L0deviceptr>(d_num_buffers),
      reinterpret_cast<L0deviceptr>(d_result_buffer)};

  // casting each kernel parameter to be a void* device ptr itself:
  std::vector<void*> kernel_param_ptrs;
  kernel_param_ptrs.reserve(num_kernel_params);
  std::transform(h_kernel_params.begin(),
                 h_kernel_params.end(),
                 std::back_inserter(kernel_param_ptrs),
                 [](L0deviceptr& param) { return &param; });

  // launching a kernel:
  typedef void* L0function;
  auto l0_func = static_cast<L0function>(gpu_context->kernel());
  // we launch as many threadblocks as there are input buffers:
  // in other words, each input buffer is handled by a single threadblock.

  // std::unique_ptr<L0DeviceCompilationContext> gpu_context
  // auto l0_ctx = dynamic_cast<const L0CompilationContext*>(gpu_context);
  // l0::L0Kernel* kernel = l0_ctx->getNativeCode(device_id);
  // l0::L0Device* device = l0_ctx->getDevice(device_id);

  auto kernel = gpu_context->kernel();
  auto device = gpu_context->device();

  auto q = device->command_queue();
  auto q_list = device->create_command_list();
  // l0::GroupCount gc = {ko.gridDimX, ko.gridDimY, ko.gridDimZ};
  l0::GroupCount gc = {1, 1, 1024};
  // LOG(INFO) << "Launching L0 kernel with group size: {" << ko.gridDimX << ","
  //           << ko.gridDimY << "," << ko.gridDimZ << "}\n";
  // q_list->launch(kernel, kernel_param_ptrs.data(), gc); //<< here is the problem

  q_list->launch(*kernel, gc);
  q_list->submit(*q.get());

  // transfer back the results:
  l0_mgr_->copyDeviceToHost(
      gpu_result_storage->getUnderlyingBuffer(), d_result_buffer, buffer_size, device_id);

  // release the gpu memory used:
  for (auto& d_buffer : d_input_buffers) {
    l0_mgr_->freeDeviceMem(d_buffer);
  }
  l0_mgr_->freeDeviceMem(d_input_buffer_dptrs);
  l0_mgr_->freeDeviceMem(d_num_buffers);
  l0_mgr_->freeDeviceMem(d_result_buffer);
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
