/*
 * Copyright 2019 OmniSci, Inc.
 * Copyright (C) 2023 Intel Corporation
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
#include "GpuSharedMemoryTestHelpers.h"

void GpuReductionTester::codegenWrapperKernel() {
  auto i8_type = llvm::Type::getInt8Ty(context_);
  auto i64_type = llvm::Type::getInt64Ty(context_);
  auto pi8_type = traits_.globalPointerType(i8_type);
  auto pi64_type = traits_.localPointerType(i64_type);
  auto ppi8_type = traits_.globalPointerType(pi8_type);

  std::vector<llvm::Type*> input_arguments;
  input_arguments.push_back(ppi8_type);
  input_arguments.push_back(i64_type);  // num input buffers
  input_arguments.push_back(pi8_type);

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
  auto input_buffer = ir_builder.CreateLoad(pi8_type, input_buffer_gep);
  auto input_buffer_ptr =
      ir_builder.CreatePointerCast(input_buffer, pi64_type, "input_buffer_ptr");
  const auto buffer_size = ll_int(
      static_cast<int32_t>(query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)),
      context_);

  // initializing shared memory and copy input buffer into shared memory buffer:
  auto init_shared_mem_func = getFunction("init_shared_mem");
  auto smem_input_buffer_ptr = ir_builder.CreateCall(init_shared_mem_func,
                                                     {
                                                         input_buffer_ptr,
                                                         buffer_size,
                                                     },
                                                     "smem_input_buffer_ptr");

  auto output_buffer_ptr =
      ir_builder.CreatePointerCast(output_buffer, pi64_type, "output_buffer_ptr");
  // call the reduction function
  CHECK(reduction_func_);
  std::vector<llvm::Value*> reduction_args{
      output_buffer_ptr, smem_input_buffer_ptr, buffer_size};
  ir_builder.CreateCall(reduction_func_, reduction_args);
  ir_builder.CreateBr(bb_exit);

  ir_builder.SetInsertPoint(bb_exit);
  ir_builder.CreateRet(nullptr);
}

compiler::CodegenTraits get_codegen_traits(const GpuMgrPlatform p) {
  switch (p) {
    case GpuMgrPlatform::CUDA:
      return compiler::CodegenTraits::get(compiler::cuda_cgen_traits_desc);
    case GpuMgrPlatform::L0:
      return compiler::CodegenTraits::get(compiler::l0_cgen_traits_desc);

    default:
      throw std::runtime_error("Unsupported GPU platform");
  }
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
                        0,
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
