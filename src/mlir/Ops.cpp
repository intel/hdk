#include "Ops.h"

#include "Attributes.h"

#include "mlir/IR/OpImplementation.h"

namespace hdk {

void ConstantOp::build(::mlir::OpBuilder& builder,
                       ::mlir::OperationState& state,
                       mlir::Type sql_type,
                       hdk::Datum value) {
  state.addTypes(sql_type);
  auto attr = hdk::DatumAttr::get(sql_type, value);
  state.addAttribute("datum", attr);
}

}  // namespace hdk

#define GET_OP_CLASSES
#include "Ops.cpp.inc"