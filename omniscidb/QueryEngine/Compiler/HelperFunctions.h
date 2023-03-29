/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <string>
#include <unordered_set>

#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include "llvm/IR/PassManager.h"

#include "QueryEngine/CompilationOptions.h"

namespace compiler {

#define MODULE_DUMP_ENABLE 1
#ifdef MODULE_DUMP_ENABLE
#define DUMP_MODULE(MODULE, LL_NAME)      \
  {                                       \
    std::error_code ec;                   \
    llvm::raw_fd_ostream os(LL_NAME, ec); \
    MODULE->print(os, nullptr);           \
  }
#else
#define DUMP_MODULE(MODULE, LL_NAME) \
  {}
#endif

#ifdef HAVE_L0
std::string mangle_spirv_builtin(const llvm::Function& func);
#endif

void throw_parseIR_error(const llvm::SMDiagnostic& parse_error,
                         std::string src = "",
                         const bool is_gpu = false);

void verify_function_ir(const llvm::Function* func);

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* llvm_module,
                 llvm::PassBuilder& PB,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co);

}  // namespace compiler
