#pragma once

#include "AST/AST.h"

#include "mlir/IR/BuiltinOps.h"

namespace hdk {

mlir::ModuleOp mlirGen(mlir::MLIRContext& context, AST::KernelSequence& moduleAST);

}
