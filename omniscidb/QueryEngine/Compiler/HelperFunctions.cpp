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
#include <llvm/IR/IRPrintingPasses.h>
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
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Path.h"

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
    DUMP_MODULE(func->getParent(), "invalid.ll");
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

std::string legalizePassName(const llvm::StringRef pass_name) {
  std::string legal_name = pass_name.str();
  constexpr std::string_view illegal_word = "w/";
  const size_t illegal_word_pos = legal_name.find(illegal_word);
  if (illegal_word_pos != std::string::npos) {
    std::string replacement =
        "with" + legal_name.substr(illegal_word_pos + illegal_word.size());
    legal_name.replace(illegal_word_pos, replacement.size(), replacement);
  }
  return legal_name;
}

const llvm::Module* unwrapModule(llvm::Any IR) {
  if (const auto** M = llvm::any_cast<const llvm::Module*>(&IR))
    return *M;

  if (const auto** F = llvm::any_cast<const llvm::Function*>(&IR)) {
    return (*F)->getParent();
  }

  if (const auto** C = llvm::any_cast<const llvm::LazyCallGraph::SCC*>(&IR)) {
    for (const llvm::LazyCallGraph::Node& N : **C) {
      const llvm::Function& F = N.getFunction();
      return F.getParent();
    }
  }

  if (const auto** L = llvm::any_cast<const llvm::Loop*>(&IR)) {
    const llvm::Function* F = (*L)->getHeader()->getParent();
    return F->getParent();
  }
  return nullptr;
}

std::string getStrTStamp(const llvm::Module* llvm_module) {
  char t_stamp_str[100];
  const std::time_t time = llvm::sys::toTimeT(std::chrono::system_clock::now());
  std::strftime(t_stamp_str, sizeof(t_stamp_str), "%F_%T", std::localtime(&time));
  return std::string{t_stamp_str}.append(".").append(std::to_string(time % 1000));
}

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* llvm_module,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  llvm::PassInstrumentationCallbacks PIC;
  size_t pass_counter{1};
  std::string ir_dump_dir{};
  if (co.dump_llvm_ir_after_each_pass) {
    ir_dump_dir = std::string("IR_DUMPS").append(llvm::sys::path::get_separator().str());
    llvm::sys::fs::create_directory(ir_dump_dir);
    ir_dump_dir.append(getStrTStamp(llvm_module))
        .append("_")
        .append(llvm_module->getTargetTriple())
        .append("_")
        .append(std::to_string(w_unit_counter++))
        .append(llvm::sys::path::get_separator().str());
    llvm::sys::fs::create_directory(ir_dump_dir);
  }

  PIC.registerAfterPassCallback(
      [&pass_counter, &ir_dump_dir](
          llvm::StringRef PassID, llvm::Any IR, const llvm::PreservedAnalyses&) -> void {
        std::error_code ec;
        llvm::raw_fd_ostream os(ir_dump_dir + std::string("IR_AFTER_")
                                                  .append(std::to_string(pass_counter++))
                                                  .append("_")
                                                  .append(legalizePassName(PassID.str())),
                                ec);
        unwrapModule(IR)->print(os, nullptr);
      });

  llvm::PassBuilder PB(nullptr,
                       llvm::PipelineTuningOptions(),
                       llvm::None,
                       (co.dump_llvm_ir_after_each_pass == 2) ? &PIC : nullptr);
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
  if (co.dump_llvm_ir_after_each_pass) {
    llvm::StandardInstrumentations SI(false);
    SI.registerCallbacks(PIC);
    DUMP_MODULE(llvm_module, ir_dump_dir + "IR_UNOPT");
  }

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
  if (co.dump_llvm_ir_after_each_pass) {
    DUMP_MODULE(llvm_module, ir_dump_dir + "IR_OPT");
  }
}
}  // namespace compiler
