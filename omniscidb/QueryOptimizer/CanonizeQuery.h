/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace hdk::ir {

class QueryDag;

void canonizeQuery(QueryDag& dag);

}  // namespace hdk::ir
