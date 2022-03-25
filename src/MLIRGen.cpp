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
      auto val = mlirGen(kernel);
      CHECK(val);
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
  // each kernel will be created as a MLIR function
  // TODO: we probably want this function nesting, so maybe our KernelOp should inherit
  // from FunctionOp?
  hdk::KernelOp mlirGen(AST::Kernel& kernel) {
    // Generate + add projected expressions
    for (auto& expr : kernel.projected_expressions) {
      if (!mlirGen(expr)) {
        CHECK(false);
        return nullptr;
      }
    }

    static int kernel_ctr = 0;
    builder.create<hdk::KernelOp>(mlir::NameLoc::get(
        mlir::Identifier::get("kernel_" + kernel_ctr++, builder.getContext())));

#if 0
    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(generateFunction());
    if (!function) {
      return nullptr;
    }

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto& entryBlock = *function.addEntryBlock();
    std::vector<std::string> arg_names = {};
    //    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue : llvm::zip(arg_names, entryBlock.getArguments())) {
      if (mlir::failed(declare(std::get<0>(nameValue), std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(generateKernelBody(kernel))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // NOTE: should we add error code here?
    hdk::ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = llvm::dyn_cast<hdk::ReturnOp>(entryBlock.back());
    if (!returnOp) {
      //      builder.create<hdk::ReturnOp>(loc(funcAST.getProto()->loc()));
      auto location =
          mlir::FileLineColLoc::get(builder.getContext(), "kernel_sequence", 0, 0);
      builder.create<hdk::ReturnOp>(location);
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      llvm_unreachable("oops");
      // function.setType(
      //     builder.getFunctionType(function.getType().getInputs(), getType(VarType{})));
    }

    return function;
#endif
  }

#if 0
  mlir::LogicalResult generateKernelBody(AST::Kernel& kernel) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

    // Generate + add projected expressions
    for (auto& expr : kernel.projected_expressions) {
      if (!mlirGen(expr)) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }
#endif

  mlir::Value mlirGen(AST::Expr* expr) {
    auto constant_expr = dynamic_cast<AST::Constant*>(expr);
    if (constant_expr) {
      if (!mlirGen(constant_expr)) {
        return nullptr;
      }
    }
    CHECK(false);
    return nullptr;
  }

  mlir::Value mlirGen(AST::Constant* constant) {
    CHECK(false);
    return nullptr;
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

#if 0
  mlir::Value mlirGen(LiteralExprAST& lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(
        lit.getDims().begin(), lit.getDims().end(), 1, std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }
#endif

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

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
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
};

mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, AST::KernelSequence& kernels) {
  return MLIRGenImpl(context).mlirGen(kernels);
}

}  // namespace hdk
