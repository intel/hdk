//===- HDKBase.td - HDK dialect definitions ----------*- tablegen -*-===//

#ifndef HDK_BASE
#define HDK_BASE

include "mlir/IR/OpBase.td"

// Provide a definition of the 'HDK' dialect in the ODS framework so that we
// can define our operations.
def HDK_Dialect : Dialect {
  let name = "hdk";
  let cppNamespace = "hdk";

  let description = [{
    Top-level dialect for HDK.
  }];
}

#endif
