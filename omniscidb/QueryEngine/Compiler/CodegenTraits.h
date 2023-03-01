#include "QueryEngine/LLVMFunctionAttributesUtil.h"

namespace compiler {

class CodegenTraits {
  explicit CodegenTraits(unsigned local_addr_space,
                         unsigned global_addr_space,
                         llvm::CallingConv::ID calling_conv,
                         llvm::StringRef triple = "")
      : local_addr_space_(local_addr_space)
      , global_addr_space_(global_addr_space)
      , conv_(calling_conv)
      , triple_(triple) {}

  const unsigned local_addr_space_;
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
                           unsigned global_addr_space,
                           llvm::CallingConv::ID calling_conv,
                           llvm::StringRef triple = "") {
    return CodegenTraits(local_addr_space, global_addr_space, calling_conv, triple);
  }

  static CodegenTraits get(CodegenTraitsDescriptor codegen_traits_desc) {
    return CodegenTraits(codegen_traits_desc.local_addr_space_,
                         codegen_traits_desc.global_addr_space_,
                         descCallingConvToLLVM.at(codegen_traits_desc.conv_),
                         codegen_traits_desc.triple_);
  }

  static CodegenTraitsDescriptor getDescriptor(unsigned local_addr_space,
                                               unsigned global_addr_space,
                                               llvm::CallingConv::ID calling_conv,
                                               const std::string triple = "") {
    return CodegenTraitsDescriptor(local_addr_space,
                                   global_addr_space,
                                   llvmCallingConvToDesc.at(calling_conv),
                                   triple);
  }

  CodegenTraitsDescriptor getDescriptor() {
    return CodegenTraitsDescriptor(local_addr_space_,
                                   global_addr_space_,
                                   llvmCallingConvToDesc.at(conv_),
                                   triple_.str());
  }

  llvm::PointerType* localPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, local_addr_space_);
  }
  llvm::PointerType* globalPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, global_addr_space_);
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
};
}  // namespace compiler