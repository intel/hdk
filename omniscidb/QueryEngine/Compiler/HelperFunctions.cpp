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

#include "HelperFunctions.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "QueryEngine/Compiler/Exceptions.h"
#include "QueryEngine/Optimization/AnnotateInternalFunctionsPass.h"

#ifdef HAVE_L0
#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#endif

namespace compiler {
#ifdef HAVE_L0
std::string mangle_spirv_builtin(const llvm::Function& func) {
  CHECK(func.getName().startswith("__spirv_")) << func.getName().str();
  std::string new_name;
  mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(), new_name);
  return new_name;
}
#endif

void throw_parseIR_error(const llvm::SMDiagnostic& parse_error,
                         std::string src,
                         const bool is_gpu) {
  std::string excname = (is_gpu ? "NVVM IR ParseError: " : "LLVM IR ParseError: ");
  llvm::raw_string_ostream ss(excname);
  parse_error.print(src.c_str(), ss, false, false);
  throw ParseIRError(ss.str());
}

void verify_function_ir(const llvm::Function* func) {
  std::stringstream err_ss;
  llvm::raw_os_ostream err_os(err_ss);
  err_os << "\n-----\n";
  if (llvm::verifyFunction(*func, &err_os)) {
    err_os << "\n-----\n";
    func->print(err_os, nullptr);
    err_os << "\n-----\n";
    LOG(FATAL) << err_ss.str();
  }
}

#if defined(HAVE_CUDA) || defined(HAVE_L0) || !defined(WITH_JIT_DEBUG)
void eliminate_dead_self_recursive_funcs(
    llvm::Module& M,
    const std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F)) {
      continue;
    }
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive) {
      dead_funcs.push_back(&F);
    }
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}
#endif

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* llvm_module,
                 llvm::legacy::PassManager& pass_manager,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  // the always inliner legacy pass must always run first
  pass_manager.add(llvm::createVerifierPass());
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());

  pass_manager.add(new AnnotateInternalFunctionsPass());

  pass_manager.add(llvm::createSROAPass());
  // mem ssa drops unused load and store instructions, e.g. passing variables directly
  // where possible
  pass_manager.add(
      llvm::createEarlyCSEPass(/*enable_mem_ssa=*/true));  // Catch trivial redundancies

  if (!is_gpu_smem_used) {
    // thread jumps can change the execution order around SMEM sections guarded by
    // `__syncthreads()`, which results in race conditions. For now, disable jump
    // threading for shared memory queries. In the future, consider handling shared memory
    // aggregations with a separate kernel launch
    pass_manager.add(llvm::createJumpThreadingPass());  // Thread jumps.
  }
  pass_manager.add(llvm::createCFGSimplificationPass());

  // remove load/stores in PHIs if instructions can be accessed directly post thread jumps
  pass_manager.add(llvm::createNewGVNPass());

  pass_manager.add(llvm::createDeadStoreEliminationPass());
  pass_manager.add(llvm::createLICMPass());

  pass_manager.add(llvm::createInstructionCombiningPass());

  // module passes
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());

  pass_manager.add(llvm::createCFGSimplificationPass());  // cleanup after everything

  pass_manager.run(*llvm_module);

  eliminate_dead_self_recursive_funcs(*llvm_module, live_funcs);
}

void insert_declaration(llvm::Module* from, llvm::Module* to, const std::string& fname) {
  auto fn = from->getFunction(fname);
  CHECK(fn);

  llvm::Function::Create(
      fn->getFunctionType(), llvm::GlobalValue::ExternalLinkage, fn->getName(), *to);
}

void replace_function(llvm::Module* from, llvm::Module* to, const std::string& fname) {
  auto target_fn = to->getFunction(fname);
  auto from_fn = from->getFunction(fname);
  CHECK(target_fn);
  CHECK(from_fn);
  CHECK(!from_fn->isDeclaration());

  target_fn->deleteBody();

  llvm::ValueToValueMapTy vmap;
  llvm::Function::arg_iterator pos_fn_arg_it = target_fn->arg_begin();
  for (llvm::Function::const_arg_iterator j = from_fn->arg_begin();
       j != from_fn->arg_end();
       ++j) {
    pos_fn_arg_it->setName(j->getName());
    vmap[&*j] = &*pos_fn_arg_it++;
  }
  llvm::SmallVector<llvm::ReturnInst*, 8> returns;
#if LLVM_VERSION_MAJOR > 12
  llvm::CloneFunctionInto(
      target_fn, from_fn, vmap, llvm::CloneFunctionChangeType::DifferentModule, returns);
#else
  llvm::CloneFunctionInto(target_fn, from_fn, vmap, true, returns);
#endif

  for (auto& BB : *target_fn) {
    for (llvm::BasicBlock::iterator bbi = BB.begin(); bbi != BB.end();) {
      llvm::Instruction* inst = &*bbi++;
      if (auto* call = llvm::dyn_cast<llvm::CallInst>(&*inst)) {
        auto local_callee = to->getFunction(call->getCalledFunction()->getName());
        CHECK(local_callee);
        std::vector<llvm::Value*> args;
        std::copy(call->arg_begin(), call->arg_end(), std::back_inserter(args));

        auto new_call = llvm::CallInst::Create(local_callee, args, call->getName());

        llvm::ReplaceInstWithInst(call, new_call);
      }
    }
  }
}

}  // namespace compiler
