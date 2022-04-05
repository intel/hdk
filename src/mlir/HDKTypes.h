#pragma once

#include "Shared/sqltypes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace hdk {
#if 0
/// Container for Datum union w/ helpers
class DatumType : public mlir::Type {
 public:
  using Type::Type;
  //  Datum() {}

  static Datum get() {}

  ::Datum datum_;
};
#endif
}  // namespace hdk