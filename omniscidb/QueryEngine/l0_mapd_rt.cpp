#include "MurmurHash.cpp"
#include "RuntimeFunctions.h"

/**
 * OCL does not provide memory-related functions due to its static nature.
 * Re-implementing them here for generic memory type to use the same code for all
 * backends.
 */

GENERIC_ADDR_SPACE void* memcpy(GENERIC_ADDR_SPACE void* dest,
                                GENERIC_ADDR_SPACE const void* src,
                                std::size_t count) {
  GENERIC_ADDR_SPACE int8_t* i8dst = reinterpret_cast<GENERIC_ADDR_SPACE int8_t*>(dest);
  const GENERIC_ADDR_SPACE int8_t* i8src =
      reinterpret_cast<const GENERIC_ADDR_SPACE int8_t*>(src);

  for (size_t i = 0; i < count; ++i) {
    i8dst[i] = i8src[i];
  }
  return dest;
}

int memcmp(GENERIC_ADDR_SPACE const void* lhs,
           GENERIC_ADDR_SPACE const void* rhs,
           std::size_t count) {
  const GENERIC_ADDR_SPACE int8_t* i8lhs =
      reinterpret_cast<const GENERIC_ADDR_SPACE int8_t*>(lhs);
  const GENERIC_ADDR_SPACE int8_t* i8rhs =
      reinterpret_cast<const GENERIC_ADDR_SPACE int8_t*>(rhs);

  for (size_t i = 0; i < count; ++i) {
    if (i8lhs[i] < i8rhs[i]) {
      return -1;
    }
    if (i8lhs[i] > i8rhs[i]) {
      return 1;
    }
  }
  return 0;
}

#include "RuntimeFunctions.cpp"
