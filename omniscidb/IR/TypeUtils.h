/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Type.h"

namespace hdk::ir {

/**
 * Return a logical type for the given type. That assumes following
 * modifications to the type (if applicable):
 *  - size for datetime types is set to 8 bytes
 *  - date type time unit is set to kSecond
 *  - external dictionary type size is set to 4 bytes
 *  - fixed-length arrays are transformed into variable-length arrays
 */
const Type* logicalType(const Type* type);

/**
 * Logical size of a type is a size of its logical type except for
 * fixed length arrays whose logical size is their original size.
 */
const int logicalSize(const Type* type);

}  // namespace hdk::ir
