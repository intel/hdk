#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

namespace hdk {

class HDKDialect : public mlir::Dialect {
 public:
  explicit HDKDialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "hdk"; }

  void initialize();
};

}