/*
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryEngine/ExpressionRewrite.h"

#include <algorithm>
#include <boost/locale/conversion.hpp>
#include <unordered_set>

#include "Analyzer/Analyzer.h"
#include "IR/ExprRewriter.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/ScalarExprVisitor.h"
#include "QueryEngine/WindowExpressionRewrite.h"
#include "Shared/sqldefs.h"

namespace {

class OrToInVisitor : public ScalarExprVisitor<std::shared_ptr<const hdk::ir::InValues>> {
 protected:
  std::shared_ptr<const hdk::ir::InValues> visitBinOper(
      const hdk::ir::BinOper* bin_oper) const override {
    switch (bin_oper->opType()) {
      case hdk::ir::OpType::kEq: {
        const auto rhs_owned = bin_oper->rightOperandShared();
        auto rhs_no_cast = extract_cast_arg(rhs_owned.get());
        if (!dynamic_cast<const hdk::ir::Constant*>(rhs_no_cast)) {
          return nullptr;
        }
        const auto arg = bin_oper->leftOperandShared();
        auto arg_type = arg->type();
        auto rhs = rhs_no_cast->cast(arg_type);
        return hdk::ir::makeExpr<hdk::ir::InValues>(arg,
                                                    std::list<hdk::ir::ExprPtr>{rhs});
      }
      case hdk::ir::OpType::kOr: {
        return aggregateResult(visit(bin_oper->leftOperand()),
                               visit(bin_oper->rightOperand()));
      }
      default:
        break;
    }
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitUOper(
      const hdk::ir::UOper* uoper) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitInValues(
      const hdk::ir::InValues*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitInIntegerSet(
      const hdk::ir::InIntegerSet*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitCharLength(
      const hdk::ir::CharLengthExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitKeyForString(
      const hdk::ir::KeyForStringExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitSampleRatio(
      const hdk::ir::SampleRatioExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitCardinality(
      const hdk::ir::CardinalityExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitLikeExpr(
      const hdk::ir::LikeExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitRegexpExpr(
      const hdk::ir::RegexpExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitCaseExpr(
      const hdk::ir::CaseExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitDateTruncExpr(
      const hdk::ir::DateTruncExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitDateDiffExpr(
      const hdk::ir::DateDiffExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitDateAddExpr(
      const hdk::ir::DateAddExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitExtractExpr(
      const hdk::ir::ExtractExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitLikelihood(
      const hdk::ir::LikelihoodExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> visitAggExpr(
      const hdk::ir::AggExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<const hdk::ir::InValues> aggregateResult(
      const std::shared_ptr<const hdk::ir::InValues>& lhs,
      const std::shared_ptr<const hdk::ir::InValues>& rhs) const override {
    if (!lhs || !rhs) {
      return nullptr;
    }

    if (lhs->arg()->type()->equal(rhs->arg()->type()) && (*lhs->arg() == *rhs->arg())) {
      auto union_values = lhs->valueList();
      const auto& rhs_values = rhs->valueList();
      union_values.insert(union_values.end(), rhs_values.begin(), rhs_values.end());
      return hdk::ir::makeExpr<hdk::ir::InValues>(lhs->argShared(), union_values);
    }
    return nullptr;
  }
};

class RecursiveOrToInVisitor : public hdk::ir::ExprRewriter {
 protected:
  hdk::ir::ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) override {
    OrToInVisitor simple_visitor;
    if (bin_oper->isOr()) {
      auto rewritten = simple_visitor.visit(bin_oper);
      if (rewritten) {
        return rewritten;
      }
    }
    auto lhs = bin_oper->leftOperandShared();
    auto rhs = bin_oper->rightOperandShared();
    auto rewritten_lhs = visit(lhs.get());
    auto rewritten_rhs = visit(rhs.get());
    return hdk::ir::makeExpr<hdk::ir::BinOper>(bin_oper->type(),
                                               bin_oper->containsAgg(),
                                               bin_oper->opType(),
                                               bin_oper->qualifier(),
                                               rewritten_lhs ? rewritten_lhs : lhs,
                                               rewritten_rhs ? rewritten_rhs : rhs);
  }
};

class ArrayElementStringLiteralEncodingVisitor : public hdk::ir::ExprRewriter {
 protected:
  hdk::ir::ExprPtr visitArrayOper(const hdk::ir::ArrayExpr* array_expr) override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < array_expr->elementCount(); ++i) {
      auto const element_expr_ptr = visit(array_expr->element(i));
      auto element_expr_type = element_expr_ptr->type();

      if (!element_expr_type->isString()) {
        args_copy.push_back(element_expr_ptr);
      } else {
        auto transient_dict_type =
            element_expr_type->ctx().extDict(element_expr_type, TRANSIENT_DICT_ID);
        args_copy.push_back(element_expr_ptr->cast(transient_dict_type));
      }
    }

    auto type = array_expr->type();
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(
        type, args_copy, array_expr->isNull(), array_expr->isLocalAlloc());
  }
};

class ConstantFoldingVisitor : public hdk::ir::ExprRewriter {
  template <typename T>
  bool foldComparison(hdk::ir::OpType optype, T t1, T t2) const {
    switch (optype) {
      case hdk::ir::OpType::kEq:
        return t1 == t2;
      case hdk::ir::OpType::kNe:
        return t1 != t2;
      case hdk::ir::OpType::kLt:
        return t1 < t2;
      case hdk::ir::OpType::kLe:
        return t1 <= t2;
      case hdk::ir::OpType::kGt:
        return t1 > t2;
      case hdk::ir::OpType::kGe:
        return t1 >= t2;
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  template <typename T>
  bool foldLogic(hdk::ir::OpType optype, T t1, T t2) const {
    switch (optype) {
      case hdk::ir::OpType::kAnd:
        return t1 && t2;
      case hdk::ir::OpType::kOr:
        return t1 || t2;
      case hdk::ir::OpType::kNot:
        return !t1;
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  // Arithmetic operates with immediate values only
  template <typename T>
  T foldArithmetic(hdk::ir::OpType optype, T t1, T t2) const {
    bool t2_is_zero = (t2 == (t2 - t2));
    bool t2_is_negative = (t2 < (t2 - t2));
    switch (optype) {
      case hdk::ir::OpType::kPlus:
        // The MIN limit for float and double is the smallest representable value,
        // not the lowest negative value! Switching to C++11 lowest.
        if ((t2_is_negative && t1 < std::numeric_limits<T>::lowest() - t2) ||
            (!t2_is_negative && t1 > std::numeric_limits<T>::max() - t2)) {
          num_overflows_++;
          throw std::runtime_error("Plus overflow");
        }
        return t1 + t2;
      case hdk::ir::OpType::kMinus:
        if ((t2_is_negative && t1 > std::numeric_limits<T>::max() + t2) ||
            (!t2_is_negative && t1 < std::numeric_limits<T>::lowest() + t2)) {
          num_overflows_++;
          throw std::runtime_error("Minus overflow");
        }
        return t1 - t2;
      case hdk::ir::OpType::kMul: {
        if (t2_is_zero) {
          return t2;
        }
        auto ct1 = t1;
        auto ct2 = t2;
        // Need to keep t2's sign on the left
        if (t2_is_negative) {
          if (t1 == std::numeric_limits<T>::lowest() ||
              t2 == std::numeric_limits<T>::lowest()) {
            // negation could overflow - bail
            num_overflows_++;
            throw std::runtime_error("Mul neg overflow");
          }
          ct1 = -t1;  // ct1 gets t2's negativity
          ct2 = -t2;  // ct2 is now positive
        }
        // Don't check overlow if we are folding FP mul by a fraction
        bool ct2_is_fraction = (ct2 < (ct2 / ct2));
        if (!ct2_is_fraction) {
          if (ct1 > std::numeric_limits<T>::max() / ct2 ||
              ct1 < std::numeric_limits<T>::lowest() / ct2) {
            num_overflows_++;
            throw std::runtime_error("Mul overflow");
          }
        }
        return t1 * t2;
      }
      case hdk::ir::OpType::kDiv:
        if (t2_is_zero) {
          throw std::runtime_error("Will not fold division by zero");
        }
        return t1 / t2;
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
  }

  bool foldOper(hdk::ir::OpType optype,
                const hdk::ir::Type* type,
                Datum lhs,
                Datum rhs,
                Datum& result,
                const hdk::ir::Type*& result_type) const {
    auto& ctx = type->ctx();
    result_type = type;

    try {
      switch (type->id()) {
        case hdk::ir::Type::kBoolean:
          if (hdk::ir::isComparison(optype)) {
            result.boolval = foldComparison<bool>(optype, lhs.boolval, rhs.boolval);
            result_type = ctx.boolean();
            return true;
          }
          if (hdk::ir::isLogic(optype)) {
            result.boolval = foldLogic<bool>(optype, lhs.boolval, rhs.boolval);
            result_type = ctx.boolean();
            return true;
          }
          CHECK(!hdk::ir::isArithmetic(optype));
          break;
        case hdk::ir::Type::kInteger:
        case hdk::ir::Type::kDecimal:
          switch (type->size()) {
            case 1:
              if (hdk::ir::isComparison(optype)) {
                result.boolval =
                    foldComparison<int8_t>(optype, lhs.tinyintval, rhs.tinyintval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.tinyintval =
                    foldArithmetic<int8_t>(optype, lhs.tinyintval, rhs.tinyintval);
                result_type = ctx.int8();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            case 2:
              if (hdk::ir::isComparison(optype)) {
                result.boolval =
                    foldComparison<int16_t>(optype, lhs.smallintval, rhs.smallintval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.smallintval =
                    foldArithmetic<int16_t>(optype, lhs.smallintval, rhs.smallintval);
                result_type = ctx.int16();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            case 4:
              if (hdk::ir::isComparison(optype)) {
                result.boolval = foldComparison<int32_t>(optype, lhs.intval, rhs.intval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.intval = foldArithmetic<int32_t>(optype, lhs.intval, rhs.intval);
                result_type = ctx.int32();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            case 8:
              if (hdk::ir::isComparison(optype)) {
                result.boolval =
                    foldComparison<int64_t>(optype, lhs.bigintval, rhs.bigintval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.bigintval =
                    foldArithmetic<int64_t>(optype, lhs.bigintval, rhs.bigintval);
                result_type = ctx.int64();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            default:
              break;
          }
          break;
        case hdk::ir::Type::kFloatingPoint:
          switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
            case hdk::ir::FloatingPointType::kFloat:
              if (hdk::ir::isComparison(optype)) {
                result.boolval =
                    foldComparison<float>(optype, lhs.floatval, rhs.floatval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.floatval =
                    foldArithmetic<float>(optype, lhs.floatval, rhs.floatval);
                result_type = ctx.fp32();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            case hdk::ir::FloatingPointType::kDouble:
              if (hdk::ir::isComparison(optype)) {
                result.boolval =
                    foldComparison<double>(optype, lhs.doubleval, rhs.doubleval);
                result_type = ctx.boolean();
                return true;
              }
              if (hdk::ir::isArithmetic(optype)) {
                result.doubleval =
                    foldArithmetic<double>(optype, lhs.doubleval, rhs.doubleval);
                result_type = ctx.fp64();
                return true;
              }
              CHECK(!hdk::ir::isLogic(optype));
              break;
            default:
              break;
          }
          break;
        default:
          break;
      }
    } catch (...) {
      return false;
    }
    return false;
  }

  hdk::ir::ExprPtr visitUOper(const hdk::ir::UOper* uoper) override {
    const auto unvisited_operand = uoper->operand();
    const auto optype = uoper->opType();
    auto type = uoper->type();
    if (optype == hdk::ir::OpType::kCast) {
      // Cache the cast type so it could be used in operand rewriting/folding
      casts_.insert({unvisited_operand, type});
    }
    const auto operand = visit(unvisited_operand);

    auto operand_type = operand->type();
    const auto const_operand =
        std::dynamic_pointer_cast<const hdk::ir::Constant>(operand);

    if (const_operand) {
      const auto operand_datum = const_operand->value();
      Datum zero_datum = {};
      Datum result_datum = {};
      const hdk::ir::Type* result_type;
      switch (optype) {
        case hdk::ir::OpType::kNot: {
          if (foldOper(hdk::ir::OpType::kEq,
                       operand_type,
                       zero_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            CHECK(result_type->isBoolean());
            return hdk::ir::makeExpr<hdk::ir::Constant>(result_type, false, result_datum);
          }
          break;
        }
        case hdk::ir::OpType::kUMinus: {
          if (foldOper(hdk::ir::OpType::kMinus,
                       operand_type,
                       zero_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            if (!operand_type->isDecimal()) {
              return hdk::ir::makeExpr<hdk::ir::Constant>(
                  result_type, false, result_datum);
            }
            return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, result_datum);
          }
          break;
        }
        case hdk::ir::OpType::kCast: {
          // Trying to fold number to number casts only
          if (!type->isNumber() || !operand_type->isNumber()) {
            break;
          }
          // Disallowing folding of FP to DECIMAL casts for now:
          // allowing them would make this test pass:
          //    update dectest set d=cast( 1234.0 as float );
          // which is expected to throw in Update.ImplicitCastToNumericTypes
          // due to cast codegen currently not supporting these casts
          if (type->isDecimal() && operand_type->isFloatingPoint()) {
            break;
          }
          auto cast_operand = const_operand->cast(type);
          auto const_cast_operand =
              std::dynamic_pointer_cast<const hdk::ir::Constant>(cast_operand);
          if (const_cast_operand) {
            auto const_cast_datum = const_cast_operand->value();
            return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, const_cast_datum);
          }
        }
        default:
          break;
      }
    }

    return hdk::ir::makeExpr<hdk::ir::UOper>(
        uoper->type(), uoper->containsAgg(), optype, operand);
  }

  hdk::ir::ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) override {
    const auto optype = bin_oper->opType();
    auto type = bin_oper->type();
    auto& ctx = type->ctx();
    auto left_operand = bin_oper->leftOperandShared();
    auto right_operand = bin_oper->rightOperandShared();

    // Check if bin_oper result is cast to a larger int or fp type
    if (casts_.find(bin_oper) != casts_.end()) {
      auto cast_type = casts_[bin_oper];
      auto lhs_type = bin_oper->leftOperand()->type();
      // Propagate cast down to the operands for folding
      if ((cast_type->isInteger() || cast_type->isFloatingPoint()) &&
          lhs_type->isInteger() && cast_type->size() > lhs_type->size() &&
          (optype == hdk::ir::OpType::kMinus || optype == hdk::ir::OpType::kPlus ||
           optype == hdk::ir::OpType::kMul)) {
        // Before folding, cast the operands to the bigger type to avoid overflows.
        // Currently upcasting smaller integer types to larger integers or double.
        left_operand = left_operand->cast(cast_type);
        right_operand = right_operand->cast(cast_type);
        type = cast_type;
      }
    }

    const auto lhs = visit(left_operand.get());
    const auto rhs = visit(right_operand.get());

    auto const_lhs = lhs->as<hdk::ir::Constant>();
    auto const_rhs = rhs->as<hdk::ir::Constant>();
    auto lhs_type = lhs->type();
    auto rhs_type = rhs->type();

    if (const_lhs && const_rhs && lhs_type->id() == rhs_type->id() && lhs_type->size() &&
        rhs_type->size()) {
      auto lhs_datum = const_lhs->value();
      auto rhs_datum = const_rhs->value();
      Datum result_datum = {};
      const hdk::ir::Type* result_type;
      if (foldOper(optype, lhs_type, lhs_datum, rhs_datum, result_datum, result_type)) {
        // Fold all ops that don't take in decimal operands, and also decimal comparisons
        if (!lhs_type->isDecimal() || hdk::ir::isComparison(optype)) {
          return hdk::ir::makeExpr<hdk::ir::Constant>(result_type, false, result_datum);
        }
        // Decimal arithmetic has been done as kBIGINT. Selectively fold some decimal ops,
        // using result_datum and BinOper expr typeinfo which was adjusted for these ops.
        if (optype == hdk::ir::OpType::kMinus || optype == hdk::ir::OpType::kPlus ||
            optype == hdk::ir::OpType::kMul) {
          return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, result_datum);
        }
      }
    }

    if (optype == hdk::ir::OpType::kAnd && lhs_type == rhs_type &&
        lhs_type->isBoolean()) {
      if (const_rhs && !const_rhs->isNull()) {
        auto rhs_datum = const_rhs->value();
        if (rhs_datum.boolval == false) {
          Datum d;
          d.boolval = false;
          // lhs && false --> false
          return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
        }
        // lhs && true --> lhs
        return lhs;
      }
      if (const_lhs && !const_lhs->isNull()) {
        auto lhs_datum = const_lhs->value();
        if (lhs_datum.boolval == false) {
          Datum d;
          d.boolval = false;
          // false && rhs --> false
          return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
        }
        // true && rhs --> rhs
        return rhs;
      }
    }
    if (optype == hdk::ir::OpType::kOr && lhs_type == rhs_type && lhs_type->isBoolean()) {
      if (const_rhs && !const_rhs->isNull()) {
        auto rhs_datum = const_rhs->value();
        if (rhs_datum.boolval == true) {
          Datum d;
          d.boolval = true;
          // lhs || true --> true
          return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
        }
        // lhs || false --> lhs
        return lhs;
      }
      if (const_lhs && !const_lhs->isNull()) {
        auto lhs_datum = const_lhs->value();
        if (lhs_datum.boolval == true) {
          Datum d;
          d.boolval = true;
          // true || rhs --> true
          return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
        }
        // false || rhs --> rhs
        return rhs;
      }
    }
    // If any value is null (N/A) result value of any op is also null (N/A)
    if (*lhs == *rhs && !lhs->type()->nullable()) {
      // Tautologies: v=v; v<=v; v>=v
      if (optype == hdk::ir::OpType::kEq || optype == hdk::ir::OpType::kLe ||
          optype == hdk::ir::OpType::kGe) {
        Datum d;
        d.boolval = true;
        return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
      }
      // Contradictions: v!=v; v<v; v>v
      if (optype == hdk::ir::OpType::kNe || optype == hdk::ir::OpType::kLt ||
          optype == hdk::ir::OpType::kGt) {
        Datum d;
        d.boolval = false;
        return hdk::ir::makeExpr<hdk::ir::Constant>(ctx.boolean(), false, d);
      }
      // v-v
      if (optype == hdk::ir::OpType::kMinus) {
        Datum d = {};
        return hdk::ir::makeExpr<hdk::ir::Constant>(lhs_type, false, d);
      }
    }
    // Convert fp division by a constant to multiplication by 1/constant
    if (optype == hdk::ir::OpType::kDiv && const_rhs && rhs_type->isFloatingPoint()) {
      auto rhs_datum = const_rhs->value();
      hdk::ir::ExprPtr recip_rhs = nullptr;
      if (rhs_type->isFp32()) {
        if (rhs_datum.floatval == 1.0) {
          return lhs;
        }
        auto f = std::fabs(rhs_datum.floatval);
        if (f > 1.0 || (f != 0.0 && 1.0 < f * std::numeric_limits<float>::max())) {
          rhs_datum.floatval = 1.0 / rhs_datum.floatval;
          recip_rhs = hdk::ir::makeExpr<hdk::ir::Constant>(rhs_type, false, rhs_datum);
        }
      } else if (rhs_type->isFp64()) {
        if (rhs_datum.doubleval == 1.0) {
          return lhs;
        }
        auto d = std::fabs(rhs_datum.doubleval);
        if (d > 1.0 || (d != 0.0 && 1.0 < d * std::numeric_limits<double>::max())) {
          rhs_datum.doubleval = 1.0 / rhs_datum.doubleval;
          recip_rhs = hdk::ir::makeExpr<hdk::ir::Constant>(rhs_type, false, rhs_datum);
        }
      }
      if (recip_rhs) {
        return hdk::ir::makeExpr<hdk::ir::BinOper>(type,
                                                   bin_oper->containsAgg(),
                                                   hdk::ir::OpType::kMul,
                                                   bin_oper->qualifier(),
                                                   lhs,
                                                   recip_rhs);
      }
    }

    return hdk::ir::makeExpr<hdk::ir::BinOper>(type,
                                               bin_oper->containsAgg(),
                                               bin_oper->opType(),
                                               bin_oper->qualifier(),
                                               lhs,
                                               rhs);
  }

  hdk::ir::ExprPtr visitLower(const hdk::ir::LowerExpr* lower_expr) override {
    const auto constant_arg_expr =
        dynamic_cast<const hdk::ir::Constant*>(lower_expr->arg());
    if (constant_arg_expr) {
      return Analyzer::analyzeStringValue(
          boost::locale::to_lower(*constant_arg_expr->value().stringval));
    }
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(lower_expr->argShared());
  }

 protected:
  mutable std::unordered_map<const hdk::ir::Expr*, const hdk::ir::Type*> casts_;
  mutable int32_t num_overflows_;

 public:
  ConstantFoldingVisitor() : num_overflows_(0) {}
  int32_t get_num_overflows() { return num_overflows_; }
  void reset_num_overflows() { num_overflows_ = 0; }
};

const hdk::ir::Expr* strip_likelihood(const hdk::ir::Expr* expr) {
  const auto with_likelihood = dynamic_cast<const hdk::ir::LikelihoodExpr*>(expr);
  if (!with_likelihood) {
    return expr;
  }
  return with_likelihood->arg();
}

}  // namespace

hdk::ir::ExprPtr rewrite_array_elements(hdk::ir::Expr const* expr) {
  return ArrayElementStringLiteralEncodingVisitor().visit(expr);
}

hdk::ir::ExprPtr rewrite_expr(const hdk::ir::Expr* expr) {
  const auto sum_window = rewrite_sum_window(expr);
  if (sum_window) {
    return sum_window;
  }
  const auto avg_window = rewrite_avg_window(expr);
  if (avg_window) {
    return avg_window;
  }
  const auto expr_no_likelihood = strip_likelihood(expr);
  // The following check is not strictly needed, but seems silly to transform a
  // simple string comparison to an IN just to codegen the same thing anyway.

  RecursiveOrToInVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  const auto expr_with_likelihood =
      std::dynamic_pointer_cast<const hdk::ir::LikelihoodExpr>(rewritten_expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<hdk::ir::LikelihoodExpr>(rewritten_expr,
                                                     expr_with_likelihood->likelihood());
  }
  return rewritten_expr;
}

hdk::ir::ExprPtr fold_expr(const hdk::ir::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto expr_no_likelihood = strip_likelihood(expr);
  ConstantFoldingVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  if (visitor.get_num_overflows() > 0 && rewritten_expr->type()->isInteger() &&
      !rewritten_expr->type()->isInt64()) {
    auto rewritten_expr_const = rewritten_expr->as<hdk::ir::Constant>();
    if (!rewritten_expr_const) {
      // Integer expression didn't fold completely the first time due to
      // overflows in smaller type subexpressions, trying again with a cast
      auto type = expr->type()->ctx().int64();
      auto bigint_expr_no_likelihood = expr_no_likelihood->cast(type);
      auto rewritten_expr_take2 = visitor.visit(bigint_expr_no_likelihood.get());
      if (rewritten_expr_take2->is<hdk::ir::Constant>()) {
        // Managed to fold, switch to the new constant
        rewritten_expr = rewritten_expr_take2;
      }
    }
  }
  const auto expr_with_likelihood = dynamic_cast<const hdk::ir::LikelihoodExpr*>(expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<hdk::ir::LikelihoodExpr>(rewritten_expr,
                                                     expr_with_likelihood->likelihood());
  }
  return rewritten_expr;
}

bool self_join_not_covered_by_left_deep_tree(const hdk::ir::ColumnVar* key_side,
                                             const hdk::ir::ColumnVar* val_side,
                                             const int max_rte_covered) {
  if (key_side->tableId() == val_side->tableId() &&
      key_side->rteIdx() == val_side->rteIdx() && key_side->rteIdx() > max_rte_covered) {
    return true;
  }
  return false;
}

const int get_max_rte_scan_table(
    std::unordered_map<int, llvm::Value*>& scan_idx_to_hash_pos) {
  int ret = INT32_MIN;
  for (auto& kv : scan_idx_to_hash_pos) {
    if (kv.first > ret) {
      ret = kv.first;
    }
  }
  return ret;
}
