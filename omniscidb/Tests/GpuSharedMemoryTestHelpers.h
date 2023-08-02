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

#include <vector>

class GpuReductionTester : public GpuSharedMemCodeBuilder {
 public:
  GpuReductionTester(const Config& config,
                     llvm::Module* module,
                     llvm::LLVMContext& context,
                     const QueryMemoryDescriptor& qmd,
                     const std::vector<TargetInfo>& targets,
                     const std::vector<int64_t>& init_agg_values,
                     GpuMgr* gpu_mgr,
                     const compiler::CodegenTraits& traits,
                     Executor* executor)
      : GpuSharedMemCodeBuilder(module,
                                context,
                                qmd,
                                targets,
                                init_agg_values,
                                config,
                                traits,
                                executor)
      , gpu_mgr_(gpu_mgr) {}
  void codegenWrapperKernel();
  llvm::Function* getWrapperKernel() const { return wrapper_kernel_; }

 protected:
  GpuMgr* gpu_mgr_;
  llvm::Function* wrapper_kernel_;
};

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

compiler::CodegenTraits get_codegen_traits(const GpuMgrPlatform p);

void init_storage_buffer(int8_t* buffer,
                         const std::vector<TargetInfo>& targets,
                         const QueryMemoryDescriptor& query_mem_desc);

std::vector<std::unique_ptr<ResultSet>> create_and_fill_input_result_sets(
    const size_t num_input_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& target_infos,
    std::vector<StrideNumberGenerator>& generators,
    const std::vector<size_t>& steps);

std::pair<std::unique_ptr<ResultSet>, std::unique_ptr<ResultSet>>
create_and_init_output_result_sets(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const std::vector<TargetInfo>& target_infos);

void perform_reduction_on_cpu(std::vector<std::unique_ptr<ResultSet>>& result_sets,
                              const ResultSetStorage* cpu_result_storage);

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
