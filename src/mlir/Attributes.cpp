#include "Attributes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "Attributes.cpp.inc"