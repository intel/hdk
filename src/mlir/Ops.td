//===- Ops.td - HDK dialect operation definitions ----------*- tablegen -*-===//

#ifndef HDK_OPS
#define HDK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

include "HDKBase.td"
include "Types.td"

// Base class for HDK dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class HDK_Op<string mnemonic, list<Trait> traits = []> :
    Op<HDK_Dialect, mnemonic, traits>;

def HDK_KernelOp : HDK_Op<"kernel"> {
    let summary = "kernel operation";
    let description = [{
        Kernels represent an individual invocation of a row-wise analytic function over some set of input columns.
        A kernel has a set layout consisting of:
            * Projected Targets
            * Groupby Targets (can we combine with above?)
            * Filter Expressions
            * Simple Filter Expressions (maybe we can determine these automagically w/ an attribute)
            * Join Expressions
            * Aggregate Expressions
            * Sort Information
    }];

    let arguments = (ins); // TODO: this could be the column / kernel arguments

    let results = (outs); // TODO: type as appropriate
    // TODO: define blocks / regions as appropriate

    let regions = (region
      AnyRegion:$projections
    );
}

def HDK_ReturnOp : HDK_Op<"return", [NoSideEffect, HasParent<"KernelOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        hdk.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  // TODO: for now just return nothing
  //  let arguments = (ins Variadic<F64Tensor>:$input);
  let arguments = (ins);

  // The return operation only emits the input in the format if it is present.
  // let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  // TODO: re-enable once/if we take a return operand
  //let builders = [
  //  OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  //];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return false; /*getNumOperands() != 0;*/ }
  }];

  // Invoke a static verify method to verify this return operation.
  // let verifier = [{ return ::verify(*this); }];
}

def HDK_ExprOp : HDK_Op<"expr", [HasParent<"::hdk::KernelOp">] > {
    let summary = "Defines an expression within a HDK kernel.";

    let description = [{
        Expressions are individual operators inside a kernel. Expressions must exist within one of the pre-defined
        kernel slots.
    }];

      let arguments = (ins StrAttr:$name, AnyType:$value); // TODO: consider constraining
      let results = (outs);
}

def HDK_ColumnOp : HDK_Op<"column", [NoSideEffect]> {
    let summary = "column";
    let description = [{
        Column operations provide containers and types for input/output
        data columns.
    }];

    let arguments = (ins AnyType: $type, AnyType: $values);
    let results = (outs); // TODO: should this mirror inputs?
}

def HDK_ConstantOp : HDK_Op<"constant", [NoSideEffect]> {
    let summary = "constant";
    let description = [{
         Constant operation turns a literal into an SSA value. The data is attached
            to the operation as an attribute. For example:

            ```mlir
              %0 = hdk.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                                : tensor<2x3xf64>
            ```
    }];

   // https://mlir.llvm.org/docs/AttributesAndTypes/
   // do we take any inputs?
   let arguments = (ins);

    // The constant operation returns a single value.
    let results = (outs HDK_AnySQLType);

    let builders = [
     OpBuilder<(ins "mlir::Type" : $sql_type, "hdk::Datum":$value)>
    ];

// Specify a parser and printer method.
// TODO
  // let parser = [{ return ::parseConstantOp(parser, result); }];
  // let printer = [{ return ::print(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  //let builders = [
  //  OpBuilder<(ins "mlir::DenseIntElementsAttr":$value), [{
  //    build($_builder, $_state, value.getType(), value);
  //  }]>,
  //];
}

#endif
