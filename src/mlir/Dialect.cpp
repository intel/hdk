#include "mlir/Dialect.h"
#include "mlir/Ops.h"

namespace hdk {

void HDKDialect::initialize() {
  addOperations<hdk::ConstantOp>();
}

}