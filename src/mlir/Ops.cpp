#include "Ops.h"
#include "mlir/IR/OpImplementation.h"

namespace hdk {

void ConstantOp::build(::mlir::OpBuilder& builder,
                       ::mlir::OperationState& state,
                       mlir::Type sql_type,
                       hdk::Datum value) {
  state.addTypes(sql_type);
  //  state.addAttribute("datum", value);
}

}  // namespace hdk

#define GET_OP_CLASSES
#include "Ops.cpp.inc"