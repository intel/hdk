/*
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

#ifndef SHARED_DICTIONARY_VALIDATOR_H
#define SHARED_DICTIONARY_VALIDATOR_H

#include <vector>

#include "Analyzer/Analyzer.h"
#include "Catalog/Catalog.h"
#include "Catalog/ColumnDescriptor.h"

namespace Parser {
class CreateTableBaseStmt;
class SharedDictionaryDef;
}  // namespace Parser

void validate_shared_dictionary(
    const Parser::CreateTableBaseStmt* stmt,
    const Analyzer::SharedDictionaryDef* shared_dict_def,
    const std::list<ColumnDescriptor>& columns,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs_so_far,
    const Catalog_Namespace::Catalog& catalog);

const Analyzer::SharedDictionaryDef compress_reference_path(
    Analyzer::SharedDictionaryDef cur_node,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs);

void validate_shared_dictionary_order(
    const Parser::CreateTableBaseStmt* stmt,
    const Analyzer::SharedDictionaryDef* shared_dict_def,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs,
    const std::list<ColumnDescriptor>& columns);
#endif  // SHARED_DICTIONARY_VALIDATOR_H
