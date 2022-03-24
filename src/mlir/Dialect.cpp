#include "mlir/Dialect.h"
#include "mlir/Ops.h"

#include "mlir/IR/Dialect.h"

#include "Dialect.cpp.inc"

namespace hdk {

void HDKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

::mlir::Type HDKDialect::parseType(::mlir::DialectAsmParser& parser) const {
  // TODO
}

/// Print a type registered to this dialect.
void HDKDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
  // TODO
}

}  // namespace hdk