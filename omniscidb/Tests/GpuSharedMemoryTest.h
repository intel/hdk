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

#pragma once

#include "GpuSharedMemoryTestHelpers.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "Shared/TargetInfo.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <memory>

class CudaReductionTester : public GpuSharedMemCodeBuilder {
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
  void performReductionTest(const std::vector<std::unique_ptr<ResultSet>>& result_sets,
                            const ResultSetStorage* gpu_result_storage,
                            const size_t device_id);

 private:
  GpuMgr* gpu_mgr_;
  llvm::Function* wrapper_kernel_;
};
