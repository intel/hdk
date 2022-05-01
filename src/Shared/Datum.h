#pragma once

#include "Shared/sqltypes.h"

namespace hdk {

/// Container for Datum union w/ helpers
class Datum {
 public:
  Datum() {}

  static Datum get() {
    return Datum();
  }

  ::Datum datum_;
};

}  // namespace hdk