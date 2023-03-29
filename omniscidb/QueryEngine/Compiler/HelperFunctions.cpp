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

#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include "llvm/IR/PassManager.h"

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
                 llvm::PassBuilder& PB,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);

  llvm::PassBuilder PB;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::LoopPassManager LPM;
  llvm::CGSCCPassManager CGPM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM;
  llvm::FunctionPassManager FPM;

  // the always inliner legacy pass must always run first
  FPM.addPass(llvm::VerifierPass());
  MPM.addPass(llvm::AlwaysInlinerPass());

  CGPM.addPass(AnnotateInternalFunctionsPass());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));

  FPM.addPass(llvm::SROAPass());
  // mem ssa drops unused load and store instructions, e.g. passing variables directly
  // where possible
  FPM.addPass(llvm::EarlyCSEPass(/*enable_mem_ssa=*/true));  // Catch trivial redundancies

  if (!is_gpu_smem_used) {
    // thread jumps can change the execution order around SMEM sections guarded by
    // `__syncthreads()`, which results in race conditions. For now, disable jump
    // threading for shared memory queries. In the future, consider handling shared
    // memory
    // aggregations with a separate kernel launch
    FPM.addPass(llvm::JumpThreadingPass());  // Thread jumps.
  }
  FPM.addPass(llvm::SimplifyCFGPass());
  // remove load/stores in PHIs if instructions can be accessed directly post thread
  FPM.addPass(llvm::GVNPass());

  FPM.addPass(llvm::DSEPass());  // DeadStoreEliminationPass
  LPM.addPass(llvm::LICMPass());
  FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM), /*UseMemorySSA=*/true));

  FPM.addPass(llvm::InstCombinePass());
  FPM.addPass(llvm::PromotePass());

  MPM.addPass(llvm::GlobalOptPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  FPM.addPass(llvm::SimplifyCFGPass());  // cleanup after everything

  MPM.run(*llvm_module, MAM);

#if defined(HAVE_CUDA) || defined(HAVE_L0) || !defined(WITH_JIT_DEBUG)
  eliminate_dead_self_recursive_funcs(*llvm_module, live_funcs);
#endif
}
}  // namespace compiler
