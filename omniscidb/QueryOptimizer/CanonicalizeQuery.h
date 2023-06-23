/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace hdk::ir {

class QueryDag;

void canonicalizeQuery(QueryDag& dag);

}  // namespace hdk::ir
