#include "mlir/Dialect.h"
#include "mlir/Ops.h"
#include "mlir/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace hdk {

void HDKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
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