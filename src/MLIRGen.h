#pragma once

#include "AST/AST.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace hdk {

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, AST::KernelSequence &moduleAST);

}

