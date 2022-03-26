#include "MLIRGen.h"
#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Ops.h"

namespace hdk {

class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext& context) : builder(&context) {}

  // convert a HDK KernelSequence to MLIR Module
  mlir::ModuleOp mlirGen(AST::KernelSequence& kernels) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto& kernel : kernels) {
      //      builder.createBlock(&theModule.body());
      auto val = mlirGen(kernel);
      CHECK(val);
      // TODO: using the builder it seems we are building this dynamically
      theModule.push_back(val);
    }

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
  hdk::KernelOp mlirGen(AST::Kernel& kernel) {
    static int kernel_ctr = 0;
    auto kernel_op = builder.create<hdk::KernelOp>(mlir::NameLoc::get(
        mlir::Identifier::get("kernel_" + kernel_ctr++, builder.getContext())));

    // Generate + add projected expressions
    for (auto& expr : kernel.projected_expressions) {
      CHECK(expr);
      auto& projection_region = kernel_op.getOperation()->getRegion(0);
      projection_region.emplaceBlock();
      CHECK(projection_region.hasOneBlock());
      builder.setInsertionPointToStart(&projection_region.front());
      if (mlir::failed(mlirGen(expr.get()))) {
        CHECK(false);
        return nullptr;
      }
    }

    // add a terminator

    builder.create<hdk::ReturnOp>(mlir::NameLoc::get(
        mlir::Identifier::get("projection_returns", builder.getContext())));

    return kernel_op;
  }

  mlir::LogicalResult mlirGen(AST::Expr* expr) {
    auto constant_expr = dynamic_cast<AST::Constant*>(expr);
    if (constant_expr) {
      if (mlir::failed(mlirGen(constant_expr))) {
        return mlir::failure();
      }
      return mlir::success();
    }
    CHECK(false);
    return mlir::failure();
  }

  mlir::LogicalResult mlirGen(AST::Constant* constant) {
    auto data_type = mlir::RankedTensorType::get({}, builder.getI64Type());
    //        mlir::IntegerType::get(builder.getContext(), 64);
    auto payload = mlir::DenseElementsAttr::get(data_type, int64_t(10));
    auto sql_type = builder.getType<hdk::BigIntType>();
    builder.create<hdk::ConstantOp>(mlir::NameLoc::get(mlir::Identifier::get(
                                        constant->toString(), builder.getContext())),
                                    payload.getType(),
                                    payload);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::FuncOp generateFunction() {
    auto location =
        mlir::FileLineColLoc::get(builder.getContext(), "kernel_sequence", 0, 0);

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    //    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
    //                                               getType(VarType{}));
    auto func_type = builder.getFunctionType({}, llvm::None);
    return mlir::FuncOp::create(location, "kernel_sequence", func_type);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
#if 0
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
#endif

  /// A "module" matches a kernel sequence: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  //  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
};

mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, AST::KernelSequence& kernels) {
  return MLIRGenImpl(context).mlirGen(kernels);
}

}  // namespace hdk
