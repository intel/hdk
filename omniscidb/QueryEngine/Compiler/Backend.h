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

#include <llvm/IR/Value.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <memory>

#include "QueryEngine/ExtensionModules.h"
#include "QueryEngine/L0Kernel.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/Target.h"

#include "CodegenTraitsDescriptor.h"

namespace CudaMgr_Namespace {
enum class NvidiaDeviceArch;
}

class CudaCompilationContext;

namespace compiler {

class CodegenTraits {
  explicit CodegenTraits(unsigned local_addr_space,
                         unsigned smem_addr_space,
                         unsigned global_addr_space,
                         llvm::CallingConv::ID calling_conv,
                         llvm::StringRef triple = "")
      : local_addr_space_(local_addr_space)
      , smem_addr_space_(smem_addr_space)
      , global_addr_space_(global_addr_space)
      , conv_(calling_conv)
      , triple_(triple) {}

  const unsigned local_addr_space_;
  const unsigned smem_addr_space_;
  const unsigned global_addr_space_;
  const llvm::CallingConv::ID conv_;
  const llvm::StringRef triple_;

  static const std::unordered_map<CallingConvDesc, llvm::CallingConv::ID>
      descCallingConvToLLVM;
  static const std::unordered_map<llvm::CallingConv::ID, CallingConvDesc>
      llvmCallingConvToDesc;

 public:
  CodegenTraits(const CodegenTraits&) = delete;
  CodegenTraits& operator=(const CodegenTraits&) = delete;

  static CodegenTraits get(unsigned local_addr_space,
                           unsigned smem_addr_space,
                           unsigned global_addr_space,
                           llvm::CallingConv::ID calling_conv,
                           llvm::StringRef triple = "") {
    return CodegenTraits(
        local_addr_space, smem_addr_space, global_addr_space, calling_conv, triple);
  }

  static CodegenTraits get(CodegenTraitsDescriptor codegen_traits_desc);
  static CodegenTraitsDescriptor getDescriptor(unsigned local_addr_space,
                                               unsigned shared_addr_space,
                                               unsigned global_addr_space,
                                               llvm::CallingConv::ID calling_conv,
                                               const std::string triple = "");

  CodegenTraitsDescriptor getDescriptor() {
    return CodegenTraitsDescriptor(local_addr_space_,
                                   smem_addr_space_,
                                   global_addr_space_,
                                   llvmCallingConvToDesc.at(conv_),
                                   triple_.str());
  }

  llvm::PointerType* localPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, local_addr_space_);
  }
  llvm::PointerType* localOpaquePtr(llvm::LLVMContext& ctx) const {
    return llvm::PointerType::get(ctx, local_addr_space_);
  }
  llvm::PointerType* smemPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, smem_addr_space_);
  }
  llvm::PointerType* globalPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, global_addr_space_);
  }
  llvm::PointerType* globalOpaquePtr(llvm::LLVMContext& ctx) const {
    return llvm::PointerType::get(ctx, global_addr_space_);
  }
  llvm::CallingConv::ID callingConv() const { return conv_; }
  llvm::StringRef dataLayout() const {
    return llvm::StringRef(
        "e-p:64:64:64-i1:8:8-i8:8:8-"
        "i16:16:16-i32:32:32-i64:64:64-"
        "f32:32:32-f64:64:64-v16:16:16-"
        "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  }
  llvm::StringRef triple() const { return triple_; }
  unsigned getLocalAddrSpace() const { return local_addr_space_; }
};

class Backend {
 public:
  virtual ~Backend(){};
  virtual std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) = 0;
  CodegenTraits traits() const { return CodegenTraits::get(traitsDesc()); };
  virtual CodegenTraitsDescriptor traitsDesc() const = 0;
  virtual void setSharedMemory(bool is_gpu_smem_used) = 0;
  CodegenTraits traits(const CodegenTraitsDescriptor& codegenTraitsDesc) const {
    return CodegenTraits::get(codegenTraitsDesc);
  };
};

class CPUBackend : public Backend {
 public:
  CPUBackend() = default;
  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func /*ignored*/,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  CodegenTraitsDescriptor traitsDesc() const { return traitsDescriptor; };

  void setSharedMemory(bool is_gpu_smem_used) {
    CHECK(false) << "Unsupported Shared Memory on CPU";
  };

  static std::shared_ptr<CpuCompilationContext> generateNativeCPUCode(
      llvm::Function* func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co);

 private:
  inline const static CodegenTraitsDescriptor traitsDescriptor{cpu_cgen_traits_desc};
};

class CUDABackend : public Backend {
 public:
  CUDABackend(const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
              bool is_gpu_smem_used,
              GPUTarget& gpu_target);

  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  CodegenTraitsDescriptor traitsDesc() const { return traitsDescriptor; };

  void setSharedMemory(bool is_gpu_smem_used) { is_gpu_smem_used_ = is_gpu_smem_used; };

  static std::string generatePTX(const std::string& cuda_llir,
                                 llvm::TargetMachine* nvptx_target_machine,
                                 llvm::LLVMContext& context);

  static void linkModuleWithLibdevice(const std::unique_ptr<llvm::Module>& ext,
                                      llvm::Module& module,
                                      llvm::PassManagerBuilder& pass_manager_builder,
                                      const GPUTarget& gpu_target,
                                      llvm::TargetMachine* nvptx_target_machine);

  static std::unique_ptr<llvm::TargetMachine> initializeNVPTXBackend(
      const CudaMgr_Namespace::NvidiaDeviceArch arch);

  static std::shared_ptr<CudaCompilationContext> generateNativeGPUCode(
      const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const bool is_gpu_smem_used,
      const CompilationOptions& co,
      const GPUTarget& gpu_target,
      llvm::TargetMachine* nvptx_target_machine);

 private:
  const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts_;
  bool is_gpu_smem_used_;
  GPUTarget& gpu_target_;

  mutable std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;
  inline const static CodegenTraitsDescriptor traitsDescriptor{cuda_cgen_traits_desc};
};

class L0Backend : public Backend {
 public:
  L0Backend(const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
            GPUTarget& gpu_target)
      : gpu_target_(gpu_target), exts_(exts) {}

  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  CodegenTraitsDescriptor traitsDesc() const { return traitsDescriptor; };

  void setSharedMemory(bool is_gpu_smem_used) { is_gpu_smem_used_ = is_gpu_smem_used; };

  static std::shared_ptr<L0CompilationContext> generateNativeGPUCode(
      const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co,
      const GPUTarget& gpu_target);

 private:
  GPUTarget& gpu_target_;
  bool is_gpu_smem_used_;
  const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts_;
  inline const static CodegenTraitsDescriptor traitsDescriptor{l0_cgen_traits_desc};
};

std::shared_ptr<Backend> getBackend(
    ExecutorDeviceType dt,
    const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
    bool is_gpu_smem_used_,
    GPUTarget& gpu_target);

void setSharedMemory(ExecutorDeviceType dt,
                     bool is_gpu_smem_used_,
                     GPUTarget& gpu_target,
                     const std::shared_ptr<compiler::Backend>& backend);

}  // namespace compiler
