#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Builders.h"

#include "mlir/Types.h"

#define GET_OP_CLASSES
#include "Ops.h.inc"

