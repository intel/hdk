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

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include "QueryEngine/CompilationOptions.h"

namespace compiler {
static std::atomic<size_t> w_unit_counter{0};
static const std::string_view tstamp_module_varname{"ModuleTstamp"};

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
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co);

// Copies a function declaration between "from" and "to" modules found by name
void insert_declaration(llvm::Module* from, llvm::Module* to, const std::string& fname);

// Inserts all globals from "from" to "to" module
void insert_globals(llvm::Module* from, llvm::Module* to);

// Clones a function body in "from" module to the same function in "to" module. The "from"
// module must contain a definition, the "to" module must contain at least a declaration.
// All the calls and global variable accesses in the copied function are retargeted to
// point to those in the "to" module.
void replace_function(llvm::Module* from, llvm::Module* to, const std::string& fname);

}  // namespace compiler
