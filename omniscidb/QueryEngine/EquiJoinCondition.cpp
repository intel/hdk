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
#include "QueryEngine/EquiJoinCondition.h"

#include "Analyzer/Analyzer.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/RangeTableIndexVisitor.h"

namespace {

// Returns true iff crt and prev are both equi-join conditions on the same pair of tables.
bool can_combine_with(const hdk::ir::Expr* crt, const hdk::ir::Expr* prev) {
  const auto crt_bin = dynamic_cast<const hdk::ir::BinOper*>(crt);
  const auto prev_bin = dynamic_cast<const hdk::ir::BinOper*>(prev);
  if (!crt_bin || !prev_bin) {
    return false;
  }
  if (!crt_bin->isEquivalence() || crt_bin->qualifier() != kONE ||
      !prev_bin->isEquivalence() || prev_bin->qualifier() != kONE ||
      // We could accept a mix of kEQ and kBW_EQ, but don't bother for now.
      crt_bin->opType() != prev_bin->opType()) {
    return false;
  }
  const auto crt_inner = std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(
      remove_cast(crt_bin->get_own_right_operand()));
  const auto prev_inner = std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(
      remove_cast(prev_bin->get_own_right_operand()));
  AllRangeTableIndexVisitor visitor;
  const auto crt_outer_rte_set = visitor.visit(crt_bin->leftOperand());
  const auto prev_outer_rte_set = visitor.visit(prev_bin->leftOperand());
  // We shouldn't treat mixed nesting levels columns as a composite key tuple.
  if (crt_outer_rte_set.size() != 1 || prev_outer_rte_set.size() != 1 ||
      crt_outer_rte_set != prev_outer_rte_set) {
    return false;
  }
  if (!crt_inner || !prev_inner || crt_inner->tableId() != prev_inner->tableId() ||
      crt_inner->rteIdx() != prev_inner->rteIdx()) {
    return false;
  }
  return true;
}

std::list<hdk::ir::ExprPtr> make_composite_equals_impl(
    const std::vector<hdk::ir::ExprPtr>& crt_coalesced_quals) {
  std::list<hdk::ir::ExprPtr> join_quals;
  std::vector<hdk::ir::ExprPtr> lhs_tuple;
  std::vector<hdk::ir::ExprPtr> rhs_tuple;
  bool nullable{false};
  for (const auto& qual : crt_coalesced_quals) {
    const auto qual_binary = qual->as<hdk::ir::BinOper>();
    CHECK(qual_binary);
    nullable = nullable || qual_binary->type()->nullable();
    const auto lhs_col = remove_cast(qual_binary->get_own_left_operand());
    const auto rhs_col = remove_cast(qual_binary->get_own_right_operand());
    const auto lhs_type = lhs_col->type();
    // Coalesce cols for integers, bool, and dict encoded strings. Forces baseline hash
    // join.
    if (lhs_type->isNumber() || lhs_type->isExtDictionary() || lhs_type->isBoolean()) {
      lhs_tuple.push_back(lhs_col);
      rhs_tuple.push_back(rhs_col);
    } else {
      join_quals.push_back(qual);
    }
  }
  CHECK(!crt_coalesced_quals.empty());
  const auto first_qual = crt_coalesced_quals.front()->as<hdk::ir::BinOper>();
  CHECK(first_qual);
  CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
  if (lhs_tuple.size() > 0) {
    join_quals.push_front(std::make_shared<hdk::ir::BinOper>(
        lhs_tuple.front()->ctx().boolean(nullable),
        false,
        first_qual->opType(),
        kONE,
        lhs_tuple.size() > 1 ? std::make_shared<hdk::ir::ExpressionTuple>(lhs_tuple)
                             : lhs_tuple.front(),
        rhs_tuple.size() > 1 ? std::make_shared<hdk::ir::ExpressionTuple>(rhs_tuple)
                             : rhs_tuple.front()));
  }
  return join_quals;
}

// Create an equals expression with column tuple operands out of regular equals
// expressions.
std::list<hdk::ir::ExprPtr> make_composite_equals(
    const std::vector<hdk::ir::ExprPtr>& crt_coalesced_quals) {
  if (crt_coalesced_quals.size() == 1) {
    return {crt_coalesced_quals.front()};
  }
  return make_composite_equals_impl(crt_coalesced_quals);
}

}  // namespace

std::list<hdk::ir::ExprPtr> combine_equi_join_conditions(
    const std::list<hdk::ir::ExprPtr>& join_quals) {
  if (join_quals.empty()) {
    return {};
  }
  std::list<hdk::ir::ExprPtr> coalesced_quals;
  std::vector<hdk::ir::ExprPtr> crt_coalesced_quals;
  for (const auto& simple_join_qual : join_quals) {
    if (crt_coalesced_quals.empty()) {
      crt_coalesced_quals.push_back(simple_join_qual);
      continue;
    }
    if (crt_coalesced_quals.size() >= g_maximum_conditions_to_coalesce ||
        !can_combine_with(simple_join_qual.get(), crt_coalesced_quals.back().get())) {
      coalesced_quals.splice(coalesced_quals.end(),
                             make_composite_equals(crt_coalesced_quals));
      crt_coalesced_quals.clear();
    }
    crt_coalesced_quals.push_back(simple_join_qual);
  }
  if (!crt_coalesced_quals.empty()) {
    coalesced_quals.splice(coalesced_quals.end(),
                           make_composite_equals(crt_coalesced_quals));
  }
  return coalesced_quals;
}

std::list<hdk::ir::ExprPtr> coalesce_singleton_equi_join(
    const std::shared_ptr<const hdk::ir::BinOper>& join_qual) {
  std::vector<hdk::ir::ExprPtr> singleton_qual_list;
  singleton_qual_list.push_back(join_qual);
  return make_composite_equals_impl(singleton_qual_list);
}
