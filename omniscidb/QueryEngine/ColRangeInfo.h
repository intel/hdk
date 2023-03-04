/*
 * Copyright 2022 Intel Corporation.
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector>

#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "ResultSet/ColRangeInfo.h"

namespace hdk::ir {
class Expr;
}
class Executor;

ColRangeInfo get_expr_range_info(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<InputTableInfo>& query_infos,
                                 const hdk::ir::Expr* expr,
                                 Executor* executor);
