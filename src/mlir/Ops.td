//===- Ops.td - HDK dialect operation definitions ----------*- tablegen -*-===//

#ifndef HDK_OPS
#define HDK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

include "HDKBase.td"
include "Types.td"

// Base class for HDK dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class HDK_Op<string mnemonic, list<OpTrait> traits = []> :
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

    let results = (outs HDK_KernelType : $result);
    // TODO: define blocks / regions as appropriate
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
        Column operations provide containers and types for input/output data columns.
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

    // The constant operation takes an attribute as the only input.
    let arguments = (ins F64ElementsAttr:$value);

    // The constant operation returns a single value of TensorType.
    let results = (outs F64Tensor);

// Specify a parser and printer method.
// TODO
  // let parser = [{ return ::parseConstantOp(parser, result); }];
  // let printer = [{ return ::print(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Invoke a static verify method to verify this constant operation.
  // TODO
  //let verifier = [{ return ::verify(*this); }];
}

#endif
