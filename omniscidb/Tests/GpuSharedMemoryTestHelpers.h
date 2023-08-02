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
#pragma once

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Compiler/Backend.h"
#include "QueryEngine/GpuSharedMemoryUtils.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/ResultSetReduction.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "ResultSetTestUtils.h"
#include "Shared/GpuPlatform.h"
#include "TestHelpers.h"

#include <vector>

class StrideNumberGenerator : public NumberGenerator {
 public:
  StrideNumberGenerator(const int64_t start, const int64_t stride)
      : crt_(start), stride_(stride), start_(start) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ += stride_;
    return crt;
  }

  void reset() override { crt_ = start_; }

 private:
  int64_t crt_;
  int64_t stride_;
  int64_t start_;
};

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
