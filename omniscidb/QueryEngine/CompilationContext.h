/*
 * Copyright 2020 OmniSci, Inc.
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

#include "QueryEngine/ExecutionEngineWrapper.h"

#include <memory>

class CompilationContext {
 public:
  virtual ~CompilationContext() {}
};

class CpuCompilationContext : public CompilationContext {
 public:
  CpuCompilationContext(std::unique_ptr<ExecutionEngineWrapper>&& execution_engine)
      : execution_engine_(std::move(execution_engine)) {}

  void setFunctionPointer(llvm::Function* function) {
    func_ = execution_engine_->getPointerToFunction(function);
    CHECK(func_);
  }

  void* getPointerToFunction(llvm::Function* function) {
    return execution_engine_->getPointerToFunction(function);
  }

  void* func() const { return func_; }

 private:
  void* func_{nullptr};
  std::unique_ptr<ExecutionEngineWrapper> execution_engine_;
};
