#include "MLIRGen.h"

#include <iostream>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

namespace hdk {

class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext& context) : builder(&context) {}

    // convert a HDK KernelSequence to MLIR Module
  mlir::ModuleOp mlirGen(AST::KernelSequence &kernels) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto& kernel : kernels) {
      // TODO -- give kernel a recursive definition?
    }
#if 0
    for (auto &record : moduleAST) {
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        auto func = mlirGen(*funcAST);
        if (!func)
          return nullptr;

        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str)))
          return nullptr;
      } else {
        llvm_unreachable("unknown record type");
      }
    }

#endif
    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

 private:
  /// A "module" matches a kernel sequence: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;
};

mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context,
                              AST::KernelSequence& kernels) {
  return MLIRGenImpl(context).mlirGen(kernels);
}

}
