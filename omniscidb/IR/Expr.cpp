/**
 * Copyright 2021 OmniSci, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Expr.h"

#include "QueryEngine/DateTimeUtils.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "Shared/misc.h"
#include "Shared/sqldefs.h"

namespace hdk::ir {

namespace {

// Return dec * 10^-scale
template <typename T>
T floatFromDecimal(int64_t const dec, unsigned const scale) {
  static_assert(std::is_floating_point_v<T>);
  return static_cast<T>(dec) / shared::power10(scale);
}

// Q: Why is there a maxRound() but no minRound()?
// A: The numerical value of std::numeric_limits<int64_t>::min() is unchanged when cast
// to either float or double, but std::numeric_limits<intXX_t>::max() is incremented to
// 2^(XX-1) when cast to float/double for XX in {32,64}, which is an invalid intXX_t
// value. Thus the maximum float/double that can be cast to a valid integer type must be
// calculated directly, and not just compared to std::numeric_limits<intXX_t>::max().
template <typename FLOAT_TYPE, typename INT_TYPE>
constexpr FLOAT_TYPE maxRound() {
  static_assert(std::is_integral_v<INT_TYPE> && std::is_floating_point_v<FLOAT_TYPE>);
  constexpr int dd =
      std::numeric_limits<INT_TYPE>::digits - std::numeric_limits<FLOAT_TYPE>::digits;
  if constexpr (0 < dd) {  // NOLINT
    return static_cast<FLOAT_TYPE>(std::numeric_limits<INT_TYPE>::max() - (1ll << dd));
  } else {
    return static_cast<FLOAT_TYPE>(std::numeric_limits<INT_TYPE>::max());
  }
}

template <typename TO, typename FROM>
TO safeNarrow(FROM const from) {
  static_assert(std::is_integral_v<TO> && std::is_integral_v<FROM>);
  static_assert(sizeof(TO) < sizeof(FROM));
  if (from < static_cast<FROM>(std::numeric_limits<TO>::min()) ||
      static_cast<FROM>(std::numeric_limits<TO>::max()) < from) {
    throw std::runtime_error("Overflow or underflow");
  }
  return static_cast<TO>(from);
}

template <typename T>
T roundDecimal(int64_t n, unsigned scale) {
  static_assert(std::is_integral_v<T>);
  constexpr size_t max_scale = std::numeric_limits<uint64_t>::digits10;  // 19
  constexpr auto pow10 = shared::powersOf<uint64_t, max_scale + 1>(10);
  if (scale == 0) {
    if constexpr (sizeof(T) < sizeof(int64_t)) {  // NOLINT
      return safeNarrow<T>(n);
    } else {
      return n;
    }
  } else if (max_scale < scale) {
    return 0;  // 0.09223372036854775807 rounds to 0
  }
  uint64_t const u = std::abs(n);
  uint64_t const pow = pow10[scale];
  uint64_t div = u / pow;
  uint64_t rem = u % pow;
  div += pow / 2 <= rem;
  if constexpr (sizeof(T) < sizeof(int64_t)) {  // NOLINT
    return safeNarrow<T>(static_cast<int64_t>(n < 0 ? -div : div));
  } else {
    return n < 0 ? -div : div;
  }
}

template <typename TO, typename FROM>
TO safeRound(FROM const from) {
  static_assert(std::is_integral_v<TO> && std::is_floating_point_v<FROM>);
  constexpr FROM max_float = maxRound<FROM, TO>();
  FROM const n = std::round(from);
  if (n < static_cast<FROM>(std::numeric_limits<TO>::min()) || max_float < n) {
    throw std::runtime_error("Overflow or underflow");
  }
  return static_cast<TO>(n);
}

// Return numeric/decimal representation of from with given scale.
template <typename T>
int64_t safeScale(T from, unsigned const scale) {
  static_assert(std::is_arithmetic_v<T>);
  constexpr size_t max_scale = std::numeric_limits<int64_t>::digits10;  // 18
  constexpr auto pow10 = shared::powersOf<int64_t, max_scale + 1>(10);
  if constexpr (std::is_integral_v<T>) {  // NOLINT
    int64_t retval;
    if (scale < pow10.size()) {
#ifdef __linux__
      if (!__builtin_mul_overflow(from, pow10[scale], &retval)) {
        return retval;
      }
      // Not over flow safe.
#else
      return from * pow10[scale];
#endif
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    if (scale < pow10.size()) {
      return safeRound<int64_t>(from * pow10[scale]);
    }
  }
  if (from == 0) {
    return 0;
  }
  throw std::runtime_error("Overflow or underflow");
}

// TODO(adb): we should revisit this, as one could argue a Datum should never contain
// a null sentinel. In fact, if we bundle Datum with a null boolean ("NullableDatum"),
// the logic becomes more explicit. There are likely other bugs associated with the
// current logic -- for example, boolean is set to -128 which is likely UB
inline bool is_null_value(const Type* type, const Datum& constval) {
  switch (type->id()) {
    case Type::kNull:
      return constval.bigintval == 0;
    case Type::kBoolean:
      return constval.tinyintval == NULL_BOOLEAN;
    case Type::kInteger:
    case Type::kDecimal:
    case Type::kExtDictionary:
      switch (type->size()) {
        case 1:
          return constval.tinyintval == NULL_TINYINT;
        case 2:
          return constval.smallintval == NULL_SMALLINT;
        case 4:
          return constval.intval == NULL_INT;
        case 8:
          return constval.bigintval == NULL_BIGINT;
        default:
          UNREACHABLE();
      }
      break;
    case Type::kFloatingPoint:
      switch (type->as<FloatingPointType>()->precision()) {
        case FloatingPointType::kFloat:
          return constval.floatval == NULL_FLOAT;
        case FloatingPointType::kDouble:
          return constval.doubleval == NULL_DOUBLE;
        default:
          UNREACHABLE();
      }
      break;
    case Type::kVarChar:
    case Type::kText:
      return constval.stringval == nullptr;
    case Type::kDate:
    case Type::kTime:
    case Type::kTimestamp:
    case Type::kInterval:
      return constval.bigintval == NULL_BIGINT;
    case Type::kFixedLenArray:
    case Type::kVarLenArray:
      return constval.arrayval == nullptr;
    case Type::kColumn:
    case Type::kColumnList:
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return false;
}

bool is_expr_nullable(const Expr* expr) {
  const auto const_expr = dynamic_cast<const Constant*>(expr);
  if (const_expr) {
    return const_expr->get_is_null();
  }
  return expr->type()->nullable();
}

bool is_in_values_nullable(const ExprPtr& a, const std::list<ExprPtr>& l) {
  if (is_expr_nullable(a.get())) {
    return true;
  }
  for (const auto& v : l) {
    if (is_expr_nullable(v.get())) {
      return true;
    }
  }
  return false;
}

bool isCastAllowed(const Type* old_type, const Type* new_type) {
  // can always cast between the same type but different precision/scale/encodings
  if (old_type->id() == new_type->id()) {
    return true;
    // can always cast from or to string
  } else if (old_type->isString() || new_type->isString()) {
    return true;
    // can always cast from or to dict encoded string
  } else if (old_type->isExtDictionary() || new_type->isExtDictionary()) {
    return true;
    // can cast between numbers
  } else if (old_type->isNumber() && new_type->isNumber()) {
    return true;
    // can cast from timestamp or date to number (epoch)
  } else if ((old_type->isTimestamp() || old_type->isDate()) && new_type->isNumber()) {
    return true;
    // can cast from number (epoch) to timestamp, date, or time
  } else if (old_type->isNumber() && new_type->isDateTime()) {
    return true;
    // can cast from date to timestamp
  } else if (old_type->isDate() && new_type->isTimestamp()) {
    return true;
  } else if (old_type->isTimestamp() && new_type->isDate()) {
    return true;
  } else if (old_type->isBoolean() && new_type->isNumber()) {
    return true;
  } else if (old_type->isArray() && new_type->isArray()) {
    auto old_elem_type = static_cast<const ArrayBaseType*>(old_type)->elemType();
    auto new_elem_type = static_cast<const ArrayBaseType*>(new_type)->elemType();
    return isCastAllowed(old_elem_type, new_elem_type);
  } else if (old_type->isColumn() && new_type->isColumn()) {
    auto old_elem_type = static_cast<const ColumnType*>(old_type)->columnType();
    auto new_elem_type = static_cast<const ColumnType*>(new_type)->columnType();
    return isCastAllowed(old_elem_type, new_elem_type);
  } else if (old_type->isColumnList() && new_type->isColumnList()) {
    auto old_elem_type = static_cast<const ColumnListType*>(old_type)->columnType();
    auto new_elem_type = static_cast<const ColumnListType*>(new_type)->columnType();
    return isCastAllowed(old_elem_type, new_elem_type);
  } else {
    return false;
  }
}

}  // namespace

void OrderEntry::print() const {
  std::cout << toString() << std::endl;
}

Expr::Expr(const Type* type, bool has_agg) : type_(type), contains_agg_(has_agg) {}

void Expr::print() const {
  std::cout << toString() << std::endl;
}

void TargetEntry::print() const {
  std::cout << toString() << std::endl;
}

ExprPtr Expr::decompress() const {
  if (type_->id() == Type::kExtDictionary) {
    auto new_type = static_cast<const ExtDictionaryType*>(type_)->elemType();
    return makeExpr<UOper>(new_type, contains_agg_, kCAST, shared_from_this());
  } else if (type_->id() == Type::kDate && type_->size() != 8) {
    auto date_type = static_cast<const DateType*>(type_);
    return makeExpr<UOper>(type_->ctx().date64(TimeUnit::kSecond, date_type->nullable()),
                           contains_agg_,
                           kCAST,
                           shared_from_this());
  } else if (type_->id() == Type::kTime && type_->size() != 8) {
    auto time_type = static_cast<const TimeType*>(type_);
    return makeExpr<UOper>(type_->ctx().time64(time_type->unit(), time_type->nullable()),
                           contains_agg_,
                           kCAST,
                           shared_from_this());
  } else if (type_->id() == Type::kInterval && type_->size() != 8) {
    auto interval_type = static_cast<const TimestampType*>(type_);
    return makeExpr<UOper>(
        type_->ctx().interval64(interval_type->unit(), interval_type->nullable()),
        contains_agg_,
        kCAST,
        shared_from_this());
  }
  return shared_from_this();
}

ExprPtr Expr::cast(const Type* new_type, bool is_dict_intersection) const {
  if (type_->equal(new_type)) {
    return shared_from_this();
  }
  if (type_->id() == Type::kExtDictionary && new_type->id() == Type::kExtDictionary) {
    auto dict_id = type_->as<ExtDictionaryType>()->dictId();
    auto new_dict_id = new_type->as<ExtDictionaryType>()->dictId();
    if (dict_id == new_dict_id || dict_id == TRANSIENT_DICT(new_dict_id)) {
      return shared_from_this();
    }
  }
  if (!isCastAllowed(type_, new_type)) {
    throw std::runtime_error("Cannot cast from " + type_->toString() + " to " +
                             new_type->toString());
  }
  // @TODO(wei) temporary restriction until executor can support this.
  if (typeid(*this) != typeid(Constant) && new_type->isExtDictionary() &&
      new_type->as<ExtDictionaryType>()->dictId() <= TRANSIENT_DICT_ID) {
    if (type_->isString()) {
      throw std::runtime_error(
          "Cannot group by string columns which are not dictionary encoded.");
    }
    throw std::runtime_error(
        "Internal error: Cannot apply transient dictionary encoding to non-literal "
        "expression "
        "yet.");
  }
  return makeExpr<UOper>(new_type, contains_agg_, kCAST, shared_from_this());
}

ExprPtr Expr::withType(const Type* type) const {
  if (!type_->equal(type)) {
    auto res = deep_copy();
    const_cast<Expr*>(res.get())->type_ = type;
    return res;
  }
  return shared_from_this();
}

std::string ColumnRef::toString() const {
  std::stringstream ss;
  ss << "(ColumnRef " << node_->getIdString() << ":" << idx_ << ")";
  return ss.str();
}

Constant::~Constant() {
  if (type_->isString() && !is_null) {
    delete constval.stringval;
  }
}

ExprPtr Constant::make(const Type* type, int64_t val, bool cacheable) {
  CHECK(type->isNumber() || type->isBoolean());
  Datum datum{0};
  switch (type->id()) {
    case Type::kBoolean:
      datum.boolval = !!val;
      break;
    case Type::kInteger:
      switch (type->size()) {
        case 1:
          datum.tinyintval = static_cast<int8_t>(val);
          break;
        case 2:
          datum.smallintval = static_cast<int16_t>(val);
          break;
        case 4:
          datum.intval = static_cast<int32_t>(val);
          break;
        case 8:
          datum.bigintval = val;
          break;
        default:
          CHECK(false);
      }
      break;
    case Type::kDecimal:
      datum.bigintval = val * exp_to_scale(type->as<DecimalType>()->scale());
      break;
    case Type::kFloatingPoint:
      switch (type->as<FloatingPointType>()->precision()) {
        case FloatingPointType::kFloat:
          datum.floatval = static_cast<float>(val);
          break;
        case FloatingPointType::kDouble:
          datum.doubleval = static_cast<double>(val);
          break;
        default:
          CHECK(false);
      }
      break;
    default:
      CHECK(false);
  }
  return makeExpr<Constant>(type, false, datum, cacheable);
}

ExprPtr ColumnVar::deep_copy() const {
  return makeExpr<ColumnVar>(col_info_, rte_idx_);
}

ExprPtr ColumnVar::withType(const Type* type) const {
  if (!type_->equal(type)) {
    auto col_info = std::make_shared<ColumnInfo>(col_info_->db_id,
                                                 col_info_->table_id,
                                                 col_info_->column_id,
                                                 col_info_->name,
                                                 type,
                                                 col_info_->is_rowid);
    return makeExpr<ColumnVar>(col_info, rte_idx_);
  }
  return shared_from_this();
}

ExprPtr ExpressionTuple::deep_copy() const {
  std::vector<ExprPtr> tuple_deep_copy;
  for (const auto& column : tuple_) {
    const auto column_deep_copy = column->deep_copy();
    CHECK(column_deep_copy->is<ColumnVar>());
    tuple_deep_copy.push_back(column_deep_copy);
  }
  return makeExpr<ExpressionTuple>(tuple_deep_copy);
}

ExprPtr Var::deep_copy() const {
  return makeExpr<Var>(col_info_, rte_idx_, which_row, varno);
}

ExprPtr Var::withType(const Type* type) const {
  if (!type_->equal(type)) {
    auto col_info = std::make_shared<ColumnInfo>(col_info_->db_id,
                                                 col_info_->table_id,
                                                 col_info_->column_id,
                                                 col_info_->name,
                                                 type,
                                                 col_info_->is_rowid);
    return makeExpr<Var>(col_info, rte_idx_, which_row, varno);
  }
  return shared_from_this();
}

ExprPtr Constant::deep_copy() const {
  Datum d = constval;
  if (type_->isString() && !is_null) {
    d.stringval = new std::string(*constval.stringval);
  }
  if (type_->isArray()) {
    return makeExpr<Constant>(type_, is_null, value_list, cacheable_);
  }
  return makeExpr<Constant>(type_, is_null, d, cacheable_);
}

ExprPtr UOper::deep_copy() const {
  return makeExpr<UOper>(
      type_, contains_agg_, optype, operand->deep_copy(), is_dict_intersection_);
}

ExprPtr BinOper::deep_copy() const {
  return makeExpr<BinOper>(type_,
                           contains_agg_,
                           optype,
                           qualifier,
                           left_operand->deep_copy(),
                           right_operand->deep_copy());
}

ExprPtr RangeOper::deep_copy() const {
  return makeExpr<RangeOper>(left_inclusive_,
                             right_inclusive_,
                             left_operand_->deep_copy(),
                             right_operand_->deep_copy());
}

ExprPtr InValues::deep_copy() const {
  std::list<ExprPtr> new_value_list;
  for (auto p : value_list) {
    new_value_list.push_back(p->deep_copy());
  }
  return makeExpr<InValues>(arg->deep_copy(), new_value_list);
}

ExprPtr CharLengthExpr::deep_copy() const {
  return makeExpr<CharLengthExpr>(arg->deep_copy(), calc_encoded_length);
}

ExprPtr KeyForStringExpr::deep_copy() const {
  return makeExpr<KeyForStringExpr>(arg->deep_copy());
}

ExprPtr SampleRatioExpr::deep_copy() const {
  return makeExpr<SampleRatioExpr>(arg->deep_copy());
}

ExprPtr LowerExpr::deep_copy() const {
  return makeExpr<LowerExpr>(arg->deep_copy());
}

ExprPtr CardinalityExpr::deep_copy() const {
  return makeExpr<CardinalityExpr>(arg->deep_copy());
}

ExprPtr LikeExpr::deep_copy() const {
  return makeExpr<LikeExpr>(arg->deep_copy(),
                            like_expr->deep_copy(),
                            escape_expr ? escape_expr->deep_copy() : nullptr,
                            is_ilike,
                            is_simple);
}

ExprPtr RegexpExpr::deep_copy() const {
  return makeExpr<RegexpExpr>(arg->deep_copy(),
                              pattern_expr->deep_copy(),
                              escape_expr ? escape_expr->deep_copy() : nullptr);
}

ExprPtr WidthBucketExpr::deep_copy() const {
  return makeExpr<WidthBucketExpr>(target_value_->deep_copy(),
                                   lower_bound_->deep_copy(),
                                   upper_bound_->deep_copy(),
                                   partition_count_->deep_copy());
}

ExprPtr LikelihoodExpr::deep_copy() const {
  return makeExpr<LikelihoodExpr>(arg->deep_copy(), likelihood);
}

ExprPtr AggExpr::deep_copy() const {
  return makeExpr<AggExpr>(
      type_, aggtype, arg ? arg->deep_copy() : nullptr, is_distinct, arg1);
}

ExprPtr CaseExpr::deep_copy() const {
  std::list<std::pair<ExprPtr, ExprPtr>> new_list;
  for (auto p : expr_pair_list) {
    new_list.emplace_back(p.first->deep_copy(), p.second->deep_copy());
  }
  return makeExpr<CaseExpr>(type_,
                            contains_agg_,
                            new_list,
                            else_expr == nullptr ? nullptr : else_expr->deep_copy());
}

ExprPtr ExtractExpr::deep_copy() const {
  return makeExpr<ExtractExpr>(type_, contains_agg_, field_, from_expr_->deep_copy());
}

ExprPtr DateaddExpr::deep_copy() const {
  return makeExpr<DateaddExpr>(
      type_, field_, number_->deep_copy(), datetime_->deep_copy());
}

ExprPtr DatediffExpr::deep_copy() const {
  return makeExpr<DatediffExpr>(type_, field_, start_->deep_copy(), end_->deep_copy());
}

ExprPtr DatetruncExpr::deep_copy() const {
  return makeExpr<DatetruncExpr>(type_, contains_agg_, field_, from_expr_->deep_copy());
}

ExprPtr OffsetInFragment::deep_copy() const {
  return makeExpr<OffsetInFragment>();
}

ExprPtr WindowFunction::deep_copy() const {
  ExprPtrVector new_args;
  for (auto& expr : args_) {
    new_args.emplace_back(expr->deep_copy());
  }
  ExprPtrVector new_partition_keys;
  for (auto& expr : partition_keys_) {
    new_partition_keys.emplace_back(expr->deep_copy());
  }
  ExprPtrVector new_order_keys;
  for (auto& expr : order_keys_) {
    new_order_keys.emplace_back(expr->deep_copy());
  }
  return makeExpr<WindowFunction>(
      type_, kind_, new_args, new_partition_keys, new_order_keys, collation_);
}

ExprPtr ArrayExpr::deep_copy() const {
  ExprPtrVector new_contained_expressions;
  for (auto& expr : contained_expressions_) {
    new_contained_expressions.emplace_back(expr->deep_copy());
  }
  return makeExpr<ArrayExpr>(type_, new_contained_expressions, is_null_, local_alloc_);
}

ExprPtr Constant::cast_number(const Type* new_type) const {
  Datum new_constval = constval;
  switch (type_->id()) {
    case Type::kBoolean:
    case Type::kInteger:
    case Type::kTimestamp: {
      int64_t old_value = extract_int_type_from_datum(constval, type_);
      if (type_->id() == Type::kBoolean) {
        old_value = old_value ? 1 : 0;
      }
      switch (new_type->id()) {
        case Type::kInteger:
          switch (new_type->size()) {
            case 1:
              new_constval.tinyintval = safeNarrow<int8_t>(old_value);
              break;
            case 2:
              new_constval.smallintval = safeNarrow<int16_t>(old_value);
              break;
            case 4:
              new_constval.intval = safeNarrow<int32_t>(old_value);
              break;
            case 8:
              new_constval.bigintval = old_value;
              break;
            default:
              abort();
          }
          break;
        case Type::kTimestamp:
          new_constval.bigintval = old_value;
          break;
        case Type::kFloatingPoint:
          switch (new_type->as<FloatingPointType>()->precision()) {
            case FloatingPointType::kFloat:
              new_constval.floatval = (float)old_value;
              break;
            case FloatingPointType::kDouble:
              new_constval.doubleval = (double)old_value;
              break;
            default:
              UNREACHABLE();
          }
          break;
        case Type::kDecimal:
          switch (new_type->size()) {
            case 8:
              new_constval.bigintval =
                  safeScale(old_value, new_type->as<DecimalType>()->scale());
              break;
            default:
              UNREACHABLE();
          }
          break;
        default:
          UNREACHABLE();
      }
    } break;
    case Type::kFloatingPoint: {
      double old_value = extract_fp_type_from_datum(constval, type_);
      switch (new_type->id()) {
        case Type::kInteger:
          switch (new_type->size()) {
            case 1:
              new_constval.tinyintval = safeRound<int8_t>(old_value);
              break;
            case 2:
              new_constval.smallintval = safeRound<int16_t>(old_value);
              break;
            case 4:
              new_constval.intval = safeRound<int32_t>(old_value);
              break;
            case 8:
              new_constval.bigintval = safeRound<int64_t>(old_value);
              break;
            default:
              abort();
          }
          break;
        case Type::kTimestamp:
          new_constval.bigintval = safeRound<int64_t>(old_value);
          break;
        case Type::kFloatingPoint:
          switch (new_type->as<FloatingPointType>()->precision()) {
            case FloatingPointType::kFloat:
              new_constval.floatval = (float)old_value;
              break;
            case FloatingPointType::kDouble:
              new_constval.doubleval = (double)old_value;
              break;
            default:
              UNREACHABLE();
          }
          break;
        case Type::kDecimal:
          switch (new_type->size()) {
            case 8:
              new_constval.bigintval =
                  safeScale(old_value, new_type->as<DecimalType>()->scale());
              break;
            default:
              UNREACHABLE();
          }
          break;
        default:
          UNREACHABLE();
      }
    } break;
    case Type::kDecimal: {
      CHECK_EQ(type_->size(), 8);
      int64_t old_value = constval.bigintval;
      int64_t old_scale = type_->as<DecimalType>()->scale();
      switch (new_type->id()) {
        case Type::kInteger:
          switch (new_type->size()) {
            case 1:
              new_constval.tinyintval = roundDecimal<int8_t>(old_value, old_scale);
              break;
            case 2:
              new_constval.smallintval = roundDecimal<int16_t>(old_value, old_scale);
              break;
            case 4:
              new_constval.intval = roundDecimal<int32_t>(old_value, old_scale);
              break;
            case 8:
              new_constval.bigintval = roundDecimal<int64_t>(old_value, old_scale);
              break;
            default:
              abort();
          }
          break;
        case Type::kTimestamp:
          new_constval.bigintval = roundDecimal<int64_t>(old_value, old_scale);
          break;
        case Type::kFloatingPoint:
          switch (new_type->as<FloatingPointType>()->precision()) {
            case FloatingPointType::kFloat:
              new_constval.floatval = floatFromDecimal<float>(old_value, old_scale);
              break;
            case FloatingPointType::kDouble:
              new_constval.doubleval = floatFromDecimal<double>(old_value, old_scale);
              break;
            default:
              UNREACHABLE();
          }
          break;
        case Type::kDecimal:
          switch (new_type->size()) {
            case 8:
              new_constval.bigintval =
                  convert_decimal_value_to_scale(old_value, type_, new_type);
              break;
            default:
              UNREACHABLE();
          }
          break;
        default:
          UNREACHABLE();
      }
    } break;
    default:
      CHECK(false);
  }
  return makeExpr<Constant>(new_type, is_null, new_constval, cacheable_);
}

ExprPtr Constant::cast_string(const Type* new_type) const {
  Datum new_constval;
  if (constval.stringval) {
    new_constval.stringval = new std::string(*constval.stringval);
    if (new_type->isVarChar()) {
      auto max_length = static_cast<size_t>(new_type->as<VarCharType>()->maxLength());
      if (max_length < new_constval.stringval->length()) {
        // truncate string
        new_constval.stringval->resize(max_length);
      }
    }
  } else {
    new_constval.stringval = nullptr;
  }
  return makeExpr<Constant>(new_type, is_null, new_constval, cacheable_);
}

ExprPtr Constant::cast_from_string(const Type* new_type) const {
  Datum new_constval = StringToDatum(*constval.stringval, new_type);
  return makeExpr<Constant>(new_type, is_null, new_constval, cacheable_);
}

ExprPtr Constant::cast_to_string(const Type* str_type) const {
  const auto str_val = DatumToString(constval, type_);
  Datum new_constval;
  new_constval.stringval = new std::string(str_val);
  if (str_type->isVarChar()) {
    // truncate the string
    auto max_length = str_type->as<hdk::ir::VarCharType>()->maxLength();
    if (new_constval.stringval->length() > static_cast<size_t>(max_length)) {
      *new_constval.stringval = new_constval.stringval->substr(0, max_length);
    }
  }
  return makeExpr<Constant>(str_type, is_null, new_constval, cacheable_);
}

ExprPtr Constant::do_cast(const Type* new_type) const {
  if (type_->equal(new_type)) {
    return shared_from_this();
  }
  if (is_null && new_type->nullable()) {
  } else if ((new_type->isNumber() || new_type->isTimestamp()) &&
             (!new_type->isTimestamp() || !type_->isTimestamp()) &&
             (type_->isNumber() || type_->isTimestamp() || type_->isBoolean())) {
    return cast_number(new_type);
  } else if (new_type->isBoolean() && type_->isBoolean()) {
  } else if (new_type->isString() && type_->isString()) {
    return cast_string(new_type);
  } else if (type_->isString()) {
    return cast_from_string(new_type);
  } else if (new_type->isString()) {
    return cast_to_string(new_type);
  } else if (new_type->isDate() && type_->isDate()) {
    CHECK(type_->as<DateType>()->unit() == new_type->as<DateType>()->unit());
  } else if (new_type->isDate() && type_->isTimestamp()) {
    Datum new_constval;
    new_constval.bigintval =
        (type_->isTimestamp() && type_->as<TimestampType>()->unit() > TimeUnit::kSecond)
            ? truncate_high_precision_timestamp_to_date(
                  constval.bigintval,
                  hdk::ir::unitsPerSecond(type_->as<TimestampType>()->unit()))
            : DateTruncate(dtDAY, constval.bigintval);
    return makeExpr<Constant>(new_type, is_null, new_constval, cacheable_);
  } else if ((type_->isTimestamp() || type_->isDate()) && new_type->isTimestamp()) {
    auto old_unit = type_->as<DateTimeBaseType>()->unit();
    auto new_unit = new_type->as<DateTimeBaseType>()->unit();
    if (old_unit != new_unit) {
      Datum new_constval;
      new_constval.bigintval = DateTimeUtils::get_datetime_scaled_epoch(
          constval.bigintval, old_unit, new_unit);
      return makeExpr<Constant>(new_type, is_null, new_constval, cacheable_);
    }
  } else if (new_type->isArray() && type_->isArray()) {
    auto new_elem_type = new_type->as<ArrayBaseType>()->elemType();
    ExprPtrList new_value_list;
    for (auto& v : value_list) {
      auto c = v->as<Constant>();
      if (!c) {
        throw std::runtime_error("Invalid array cast.");
      }
      new_value_list.push_back(c->do_cast(new_elem_type));
    }
    return makeExpr<Constant>(new_type, is_null, new_value_list, cacheable_);
  } else if (get_is_null() && (new_type->isNumber() || new_type->isTime() ||
                               new_type->isString() || new_type->isBoolean())) {
  } else if (!is_null_value(type_, constval) &&
             type_->withNullable(true)->equal(new_type)) {
    CHECK(!is_null);
    // relax nullability
  } else {
    throw std::runtime_error("Cast from " + type_->toString() + " to " +
                             new_type->toString() + " not supported");
  }
  return withType(new_type);
}

void Constant::set_null_value() {
  switch (type_->id()) {
    case Type::kBoolean:
      constval.boolval = NULL_BOOLEAN;
      break;
    case Type::kInteger:
    case Type::kDecimal:
      switch (type_->size()) {
        case 1:
          constval.tinyintval = NULL_TINYINT;
          break;
        case 2:
          constval.smallintval = NULL_SMALLINT;
          break;
        case 4:
          constval.intval = NULL_INT;
          break;
        case 8:
          constval.bigintval = NULL_BIGINT;
          break;
        default:
          CHECK(false);
      }
      break;
    case Type::kFloatingPoint:
      switch (type_->as<FloatingPointType>()->precision()) {
        case FloatingPointType::kFloat:
          constval.floatval = NULL_FLOAT;
          break;
        case FloatingPointType::kDouble:
          constval.doubleval = NULL_DOUBLE;
          break;
        default:
          CHECK(false);
      }
      break;
    case Type::kTime:
    case Type::kTimestamp:
    case Type::kDate:
      constval.bigintval = NULL_BIGINT;
      break;
    case Type::kExtDictionary:
    case Type::kVarChar:
    case Type::kText:
      constval.stringval = nullptr;
      break;
    case Type::kNull:
      constval.bigintval = 0;
      break;
    case Type::kFixedLenArray:
    case Type::kVarLenArray:
      constval.arrayval = nullptr;
      break;
    default:
      CHECK(false);
  }
}

ExprPtr Constant::cast(const Type* new_type, bool is_dict_intersection) const {
  CHECK(!is_dict_intersection);
  if (is_null) {
    return makeExpr<Constant>(new_type, true, constval, cacheable_);
  }
  if (new_type->isExtDictionary()) {
    auto new_cst = do_cast(new_type->as<ExtDictionaryType>()->elemType());
    return new_cst->Expr::cast(new_type);
  }
  if ((type_->isTime() || type_->isDate()) && new_type->isNumber()) {
    // Let the codegen phase deal with casts from date/time to a number.
    return makeExpr<UOper>(new_type, contains_agg_, kCAST, shared_from_this());
  }
  return do_cast(new_type);
}

ExprPtr UOper::cast(const Type* new_type, bool is_dict_intersection) const {
  if (optype != kCAST) {
    return Expr::cast(new_type, is_dict_intersection);
  }
  if (type_->isString() && new_type->isExtDictionary()) {
    auto otype = operand->type();
    if (otype->isExtDictionary()) {
      int op_dict_id = otype->as<ExtDictionaryType>()->dictId();
      int new_dict_id = new_type->as<ExtDictionaryType>()->dictId();
      if (op_dict_id == new_dict_id || op_dict_id == TRANSIENT_DICT(new_dict_id)) {
        return operand;
      }
    }
  }
  return Expr::cast(new_type, is_dict_intersection);
}

ExprPtr CaseExpr::cast(const Type* new_type, bool is_dict_intersection) const {
  std::list<std::pair<ExprPtr, ExprPtr>> new_expr_pair_list;
  for (auto& p : expr_pair_list) {
    new_expr_pair_list.emplace_back(
        std::make_pair(p.first, p.second->cast(new_type, is_dict_intersection)));
  }

  return makeExpr<CaseExpr>(
      new_type,
      contains_agg_,
      std::move(new_expr_pair_list),
      else_expr ? else_expr->cast(new_type, is_dict_intersection) : nullptr);
}

InValues::InValues(ExprPtr a, const std::list<ExprPtr>& l)
    : Expr(a->ctx().boolean(is_in_values_nullable(a, l))), arg(a), value_list(l) {}

InIntegerSet::InIntegerSet(const std::shared_ptr<const Expr> a,
                           const std::vector<int64_t>& l,
                           const bool not_null)
    : Expr(a->ctx().boolean(!not_null)), arg(a), value_list(l) {}

bool ColumnVar::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(ColumnVar) && typeid(rhs) != typeid(Var)) {
    return false;
  }
  const ColumnVar& rhs_cv = dynamic_cast<const ColumnVar&>(rhs);
  if (rte_idx_ != -1) {
    return (tableId() == rhs_cv.tableId()) && (columnId() == rhs_cv.columnId()) &&
           (rte_idx_ == rhs_cv.rteIdx());
  }
  const Var* v = dynamic_cast<const Var*>(this);
  if (v == nullptr) {
    return false;
  }
  const Var* rv = dynamic_cast<const Var*>(&rhs);
  if (rv == nullptr) {
    return false;
  }
  return (v->whichRow() == rv->whichRow()) && (v->varNo() == rv->varNo());
}

bool ExpressionTuple::operator==(const Expr& rhs) const {
  const auto rhs_tuple = dynamic_cast<const ExpressionTuple*>(&rhs);
  if (!rhs_tuple) {
    return false;
  }
  const auto& rhs_tuple_cols = rhs_tuple->tuple();
  return expr_list_match(tuple_, rhs_tuple_cols);
}

bool Constant::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(Constant)) {
    return false;
  }
  const Constant& rhs_c = dynamic_cast<const Constant&>(rhs);
  if (!type_->equal(rhs_c.type()) || is_null != rhs_c.get_is_null()) {
    return false;
  }
  if (is_null && rhs_c.get_is_null()) {
    return true;
  }
  if (type_->isArray()) {
    return false;
  }
  return DatumEqual(constval, rhs_c.get_constval(), type_);
}

bool UOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(UOper)) {
    return false;
  }
  const UOper& rhs_uo = dynamic_cast<const UOper&>(rhs);
  return optype == rhs_uo.get_optype() && *operand == *rhs_uo.get_operand() &&
         is_dict_intersection_ == rhs_uo.is_dict_intersection_;
}

bool BinOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(BinOper)) {
    return false;
  }
  const BinOper& rhs_bo = dynamic_cast<const BinOper&>(rhs);
  return optype == rhs_bo.get_optype() && *left_operand == *rhs_bo.get_left_operand() &&
         *right_operand == *rhs_bo.get_right_operand();
}

bool RangeOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(RangeOper)) {
    return false;
  }
  const RangeOper& rhs_rg = dynamic_cast<const RangeOper&>(rhs);
  return left_inclusive_ == rhs_rg.left_inclusive_ &&
         right_inclusive_ == rhs_rg.right_inclusive_ &&
         *left_operand_ == *rhs_rg.left_operand_ &&
         *right_operand_ == *rhs_rg.right_operand_;
}

bool CharLengthExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CharLengthExpr)) {
    return false;
  }
  const CharLengthExpr& rhs_cl = dynamic_cast<const CharLengthExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg()) ||
      calc_encoded_length != rhs_cl.get_calc_encoded_length()) {
    return false;
  }
  return true;
}

bool KeyForStringExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(KeyForStringExpr)) {
    return false;
  }
  const KeyForStringExpr& rhs_cl = dynamic_cast<const KeyForStringExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg())) {
    return false;
  }
  return true;
}

bool SampleRatioExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(SampleRatioExpr)) {
    return false;
  }
  const SampleRatioExpr& rhs_cl = dynamic_cast<const SampleRatioExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg())) {
    return false;
  }
  return true;
}

bool LowerExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LowerExpr)) {
    return false;
  }

  return *arg == *dynamic_cast<const LowerExpr&>(rhs).get_arg();
}

bool CardinalityExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CardinalityExpr)) {
    return false;
  }
  const CardinalityExpr& rhs_ca = dynamic_cast<const CardinalityExpr&>(rhs);
  if (!(*arg == *rhs_ca.get_arg())) {
    return false;
  }
  return true;
}

bool LikeExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LikeExpr)) {
    return false;
  }
  const LikeExpr& rhs_lk = dynamic_cast<const LikeExpr&>(rhs);
  if (!(*arg == *rhs_lk.get_arg()) || !(*like_expr == *rhs_lk.get_like_expr()) ||
      is_ilike != rhs_lk.get_is_ilike()) {
    return false;
  }
  if (escape_expr.get() == rhs_lk.get_escape_expr()) {
    return true;
  }
  if (escape_expr != nullptr && rhs_lk.get_escape_expr() != nullptr &&
      *escape_expr == *rhs_lk.get_escape_expr()) {
    return true;
  }
  return false;
}

bool RegexpExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(RegexpExpr)) {
    return false;
  }
  const RegexpExpr& rhs_re = dynamic_cast<const RegexpExpr&>(rhs);
  if (!(*arg == *rhs_re.get_arg()) || !(*pattern_expr == *rhs_re.get_pattern_expr())) {
    return false;
  }
  if (escape_expr.get() == rhs_re.get_escape_expr()) {
    return true;
  }
  if (escape_expr != nullptr && rhs_re.get_escape_expr() != nullptr &&
      *escape_expr == *rhs_re.get_escape_expr()) {
    return true;
  }
  return false;
}

bool WidthBucketExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(WidthBucketExpr)) {
    return false;
  }
  const WidthBucketExpr& rhs_l = dynamic_cast<const WidthBucketExpr&>(rhs);
  if (!(*target_value_ == *rhs_l.get_target_value())) {
    return false;
  }
  if (!(*lower_bound_ == *rhs_l.get_lower_bound())) {
    return false;
  }
  if (!(*upper_bound_ == *rhs_l.get_upper_bound())) {
    return false;
  }
  if (!(*partition_count_ == *rhs_l.get_partition_count())) {
    return false;
  }
  return true;
}

bool LikelihoodExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LikelihoodExpr)) {
    return false;
  }
  const LikelihoodExpr& rhs_l = dynamic_cast<const LikelihoodExpr&>(rhs);
  if (!(*arg == *rhs_l.get_arg())) {
    return false;
  }
  if (likelihood != rhs_l.get_likelihood()) {
    return false;
  }
  return true;
}

bool InValues::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(InValues)) {
    return false;
  }
  const InValues& rhs_iv = dynamic_cast<const InValues&>(rhs);
  if (!(*arg == *rhs_iv.get_arg())) {
    return false;
  }
  if (value_list.size() != rhs_iv.get_value_list().size()) {
    return false;
  }
  auto q = rhs_iv.get_value_list().begin();
  for (auto p : value_list) {
    if (!(*p == **q)) {
      return false;
    }
    q++;
  }
  return true;
}

bool AggExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(AggExpr)) {
    return false;
  }
  const AggExpr& rhs_ae = dynamic_cast<const AggExpr&>(rhs);
  if (aggtype != rhs_ae.get_aggtype() || is_distinct != rhs_ae.get_is_distinct()) {
    return false;
  }
  if (arg.get() == rhs_ae.get_arg()) {
    return true;
  }
  if (arg == nullptr || rhs_ae.get_arg() == nullptr) {
    return false;
  }
  return *arg == *rhs_ae.get_arg();
}

bool CaseExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CaseExpr)) {
    return false;
  }
  const CaseExpr& rhs_ce = dynamic_cast<const CaseExpr&>(rhs);
  if (expr_pair_list.size() != rhs_ce.get_expr_pair_list().size()) {
    return false;
  }
  if ((else_expr == nullptr && rhs_ce.get_else_expr() != nullptr) ||
      (else_expr != nullptr && rhs_ce.get_else_expr() == nullptr)) {
    return false;
  }
  auto it = rhs_ce.get_expr_pair_list().cbegin();
  for (auto p : expr_pair_list) {
    if (!(*p.first == *it->first) || !(*p.second == *it->second)) {
      return false;
    }
    ++it;
  }
  return else_expr == nullptr ||
         (else_expr != nullptr && *else_expr == *rhs_ce.get_else_expr());
}

bool ExtractExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(ExtractExpr)) {
    return false;
  }
  const ExtractExpr& rhs_ee = dynamic_cast<const ExtractExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *from_expr_ == *rhs_ee.get_from_expr();
}

bool DateaddExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DateaddExpr)) {
    return false;
  }
  const DateaddExpr& rhs_ee = dynamic_cast<const DateaddExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *number_ == *rhs_ee.get_number_expr() &&
         *datetime_ == *rhs_ee.get_datetime_expr();
}

bool DatediffExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DatediffExpr)) {
    return false;
  }
  const DatediffExpr& rhs_ee = dynamic_cast<const DatediffExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *start_ == *rhs_ee.get_start_expr() &&
         *end_ == *rhs_ee.get_end_expr();
}

bool DatetruncExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DatetruncExpr)) {
    return false;
  }
  const DatetruncExpr& rhs_ee = dynamic_cast<const DatetruncExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *from_expr_ == *rhs_ee.get_from_expr();
}

bool OffsetInFragment::operator==(const Expr& rhs) const {
  return typeid(rhs) == typeid(OffsetInFragment);
}

bool WindowFunction::operator==(const Expr& rhs) const {
  const auto rhs_window = dynamic_cast<const WindowFunction*>(&rhs);
  if (!rhs_window) {
    return false;
  }
  if (kind_ != rhs_window->kind_ || args_.size() != rhs_window->args_.size() ||
      partition_keys_.size() != rhs_window->partition_keys_.size() ||
      order_keys_.size() != rhs_window->order_keys_.size()) {
    return false;
  }
  return expr_list_match(args_, rhs_window->args_) &&
         expr_list_match(partition_keys_, rhs_window->partition_keys_) &&
         expr_list_match(order_keys_, rhs_window->order_keys_);
}

bool ArrayExpr::operator==(Expr const& rhs) const {
  if (typeid(rhs) != typeid(ArrayExpr)) {
    return false;
  }
  ArrayExpr const& casted_rhs = static_cast<ArrayExpr const&>(rhs);
  for (unsigned i = 0; i < contained_expressions_.size(); i++) {
    auto& lhs_expr = contained_expressions_[i];
    auto& rhs_expr = casted_rhs.contained_expressions_[i];
    if (!(*lhs_expr == *rhs_expr)) {
      return false;
    }
  }
  if (isNull() != casted_rhs.isNull()) {
    return false;
  }

  return true;
}

std::string ColumnVar::toString() const {
  return "(ColumnVar table: " + std::to_string(tableId()) +
         " column: " + std::to_string(columnId()) + " rte: " + std::to_string(rte_idx_) +
         " " + type()->toString() + ") ";
}

std::string ExpressionTuple::toString() const {
  std::string str{"< "};
  for (const auto& column : tuple_) {
    str += column->toString();
  }
  str += "> ";
  return str;
}

std::string Var::toString() const {
  return "(Var table: " + std::to_string(tableId()) +
         " column: " + std::to_string(columnId()) + " rte: " + std::to_string(rte_idx_) +
         " which_row: " + std::to_string(which_row) + " varno: " + std::to_string(varno) +
         ") ";
}

std::string Constant::toString() const {
  std::string str{"(Const "};
  if (is_null) {
    str += "NULL";
  } else if (type_->isArray()) {
    str += type_->toString();
  } else {
    str += DatumToString(constval, type_);
  }
  str += ") ";
  return str;
}

std::string UOper::toString() const {
  std::string op;
  switch (optype) {
    case kNOT:
      op = "NOT ";
      break;
    case kUMINUS:
      op = "- ";
      break;
    case kISNULL:
      op = "IS NULL ";
      break;
    case kEXISTS:
      op = "EXISTS ";
      break;
    case kCAST:
      op = "CAST " + type_->toString() + " ";
      break;
    case kUNNEST:
      op = "UNNEST ";
      break;
    default:
      break;
  }
  std::string dict_int = is_dict_intersection_ ? " DICT_INT" : "";
  return "(" + op + operand->toString() + dict_int + ") ";
}

std::string BinOper::toString() const {
  std::string op;
  switch (optype) {
    case kEQ:
      op = "= ";
      break;
    case kNE:
      op = "<> ";
      break;
    case kLT:
      op = "< ";
      break;
    case kLE:
      op = "<= ";
      break;
    case kGT:
      op = "> ";
      break;
    case kGE:
      op = ">= ";
      break;
    case kAND:
      op = "AND ";
      break;
    case kOR:
      op = "OR ";
      break;
    case kMINUS:
      op = "- ";
      break;
    case kPLUS:
      op = "+ ";
      break;
    case kMULTIPLY:
      op = "* ";
      break;
    case kDIVIDE:
      op = "/ ";
      break;
    case kMODULO:
      op = "% ";
      break;
    case kARRAY_AT:
      op = "[] ";
      break;
    case kBW_EQ:
      op = "BW_EQ ";
      break;
    default:
      break;
  }
  std::string str{"("};
  str += op;
  if (qualifier == kANY) {
    str += "ANY ";
  } else if (qualifier == kALL) {
    str += "ALL ";
  }
  str += left_operand->toString();
  str += right_operand->toString();
  str += ") ";
  return str;
}

std::string RangeOper::toString() const {
  const std::string lhs = left_inclusive_ ? "[" : "(";
  const std::string rhs = right_inclusive_ ? "]" : ")";
  return "(RangeOper " + lhs + " " + left_operand_->toString() + " , " +
         right_operand_->toString() + " " + rhs + " )";
}

std::string ScalarSubquery::toString() const {
  return "(Subquery node: " + std::to_string(node_->getId()) + ")";
}

std::string InValues::toString() const {
  std::string str{"(IN "};
  str += arg->toString();
  str += "(";
  int cnt = 0;
  bool shorted_value_list_str = false;
  for (auto e : value_list) {
    str += e->toString();
    cnt++;
    if (cnt > 4) {
      shorted_value_list_str = true;
      break;
    }
  }
  if (shorted_value_list_str) {
    str += "... | ";
    str += "Total # values: ";
    str += std::to_string(value_list.size());
  }
  str += ") ";
  return str;
}

ExprPtr InIntegerSet::deep_copy() const {
  return std::make_shared<InIntegerSet>(
      arg->deep_copy(), value_list, !type()->nullable());
}

bool InIntegerSet::operator==(const Expr& rhs) const {
  if (!dynamic_cast<const InIntegerSet*>(&rhs)) {
    return false;
  }
  const auto& rhs_in_integer_set = static_cast<const InIntegerSet&>(rhs);
  return *arg == *rhs_in_integer_set.arg && value_list == rhs_in_integer_set.value_list;
}

std::string InIntegerSet::toString() const {
  std::string str{"(IN_INTEGER_SET "};
  str += arg->toString();
  str += "( ";
  int cnt = 0;
  bool shorted_value_list_str = false;
  for (const auto e : value_list) {
    str += std::to_string(e) + " ";
    cnt++;
    if (cnt > 4) {
      shorted_value_list_str = true;
      break;
    }
  }
  if (shorted_value_list_str) {
    str += "... | ";
    str += "Total # values: ";
    str += std::to_string(value_list.size());
  }
  str += ") ";
  return str;
}

std::string InSubquery::toString() const {
  return "(InSubquery arg: " + arg_->toString() +
         " node: " + std::to_string(node_->getId()) + ")";
}

std::string CharLengthExpr::toString() const {
  std::string str;
  if (calc_encoded_length) {
    str += "CHAR_LENGTH(";
  } else {
    str += "LENGTH(";
  }
  str += arg->toString();
  str += ") ";
  return str;
}

std::string KeyForStringExpr::toString() const {
  std::string str{"KEY_FOR_STRING("};
  str += arg->toString();
  str += ") ";
  return str;
}

std::string SampleRatioExpr::toString() const {
  std::string str{"SAMPLE_RATIO("};
  str += arg->toString();
  str += ") ";
  return str;
}

std::string LowerExpr::toString() const {
  return "LOWER(" + arg->toString() + ") ";
}

std::string CardinalityExpr::toString() const {
  std::string str{"CARDINALITY("};
  str += arg->toString();
  str += ") ";
  return str;
}

std::string LikeExpr::toString() const {
  std::string str{"(LIKE "};
  str += arg->toString();
  str += like_expr->toString();
  if (escape_expr) {
    str += escape_expr->toString();
  }
  str += ") ";
  return str;
}

std::string RegexpExpr::toString() const {
  std::string str{"(REGEXP "};
  str += arg->toString();
  str += pattern_expr->toString();
  if (escape_expr) {
    str += escape_expr->toString();
  }
  str += ") ";
  return str;
}

std::string WidthBucketExpr::toString() const {
  std::string str{"(WIDTH_BUCKET "};
  str += target_value_->toString();
  str += lower_bound_->toString();
  str += upper_bound_->toString();
  str += partition_count_->toString();
  return str + ") ";
}

std::string LikelihoodExpr::toString() const {
  std::string str{"(LIKELIHOOD "};
  str += arg->toString();
  return str + " " + std::to_string(likelihood) + ") ";
}

std::string AggExpr::toString() const {
  std::string agg;
  switch (aggtype) {
    case kAVG:
      agg = "AVG ";
      break;
    case kMIN:
      agg = "MIN ";
      break;
    case kMAX:
      agg = "MAX ";
      break;
    case kSUM:
      agg = "SUM ";
      break;
    case kCOUNT:
      agg = "COUNT ";
      break;
    case kAPPROX_COUNT_DISTINCT:
      agg = "APPROX_COUNT_DISTINCT";
      break;
    case kAPPROX_QUANTILE:
      agg = "APPROX_PERCENTILE";
      break;
    case kSINGLE_VALUE:
      agg = "SINGLE_VALUE";
      break;
    case kSAMPLE:
      agg = "SAMPLE";
      break;
  }
  std::string str{"(" + agg};
  if (is_distinct) {
    str += "DISTINCT ";
  }
  if (arg) {
    str += arg->toString();
  } else {
    str += "*";
  }
  return str + ") ";
}

std::string CaseExpr::toString() const {
  std::string str{"CASE "};
  for (auto p : expr_pair_list) {
    str += "(";
    str += p.first->toString();
    str += ", ";
    str += p.second->toString();
    str += ") ";
  }
  if (else_expr) {
    str += "ELSE ";
    str += else_expr->toString();
  }
  str += " END ";
  return str;
}

std::string ExtractExpr::toString() const {
  return "EXTRACT(" + std::to_string(field_) + " FROM " + from_expr_->toString() + ") ";
}

std::string DateaddExpr::toString() const {
  return "DATEADD(" + std::to_string(field_) + " NUMBER " + number_->toString() +
         " DATETIME " + datetime_->toString() + ") ";
}

std::string DatediffExpr::toString() const {
  return "DATEDIFF(" + std::to_string(field_) + " START " + start_->toString() + " END " +
         end_->toString() + ") ";
}

std::string DatetruncExpr::toString() const {
  return "DATE_TRUNC(" + std::to_string(field_) + " , " + from_expr_->toString() + ") ";
}

std::string OffsetInFragment::toString() const {
  return "(OffsetInFragment) ";
}

std::string WindowFunction::toString() const {
  std::string result = "WindowFunction(" + ::toString(kind_);
  for (const auto& arg : args_) {
    result += " " + arg->toString();
  }
  return result + ") ";
}

std::string ArrayExpr::toString() const {
  std::string str{"ARRAY["};

  auto iter(contained_expressions_.begin());
  while (iter != contained_expressions_.end()) {
    str += (*iter)->toString();
    if (iter + 1 != contained_expressions_.end()) {
      str += ", ";
    }
    iter++;
  }
  str += "]";
  return str;
}

std::string TargetEntry::toString() const {
  std::string str{"(" + resname + " "};
  str += expr->toString();
  if (unnest) {
    str += " UNNEST";
  }
  str += ") ";
  return str;
}

std::string OrderEntry::toString() const {
  std::string str{std::to_string(tle_no)};
  if (is_desc) {
    str += " desc";
  }
  if (nulls_first) {
    str += " nulls first";
  }
  str += " ";
  return str;
}

ExprPtr FunctionOper::deep_copy() const {
  std::vector<ExprPtr> args_copy;
  for (size_t i = 0; i < getArity(); ++i) {
    args_copy.push_back(getArg(i)->deep_copy());
  }
  return makeExpr<FunctionOper>(type_, getName(), args_copy);
}

bool FunctionOper::operator==(const Expr& rhs) const {
  if (!type_->equal(rhs.type())) {
    return false;
  }
  const auto rhs_func_oper = dynamic_cast<const FunctionOper*>(&rhs);
  if (!rhs_func_oper) {
    return false;
  }
  if (getName() != rhs_func_oper->getName()) {
    return false;
  }
  if (getArity() != rhs_func_oper->getArity()) {
    return false;
  }
  for (size_t i = 0; i < getArity(); ++i) {
    if (!(*getArg(i) == *(rhs_func_oper->getArg(i)))) {
      return false;
    }
  }
  return true;
}

std::string FunctionOper::toString() const {
  std::string str{"(" + name_ + " "};
  for (const auto& arg : args_) {
    str += arg->toString();
  }
  str += ")";
  return str;
}

ExprPtr FunctionOperWithCustomTypeHandling::deep_copy() const {
  std::vector<ExprPtr> args_copy;
  for (size_t i = 0; i < getArity(); ++i) {
    args_copy.push_back(getArg(i)->deep_copy());
  }
  return makeExpr<FunctionOperWithCustomTypeHandling>(type_, getName(), args_copy);
}

bool FunctionOperWithCustomTypeHandling::operator==(const Expr& rhs) const {
  if (!type_->equal(rhs.type())) {
    return false;
  }
  const auto rhs_func_oper =
      dynamic_cast<const FunctionOperWithCustomTypeHandling*>(&rhs);
  if (!rhs_func_oper) {
    return false;
  }
  if (getName() != rhs_func_oper->getName()) {
    return false;
  }
  if (getArity() != rhs_func_oper->getArity()) {
    return false;
  }
  for (size_t i = 0; i < getArity(); ++i) {
    if (!(*getArg(i) == *(rhs_func_oper->getArg(i)))) {
      return false;
    }
  }
  return true;
}

double WidthBucketExpr::get_bound_val(const Expr* bound_expr) const {
  CHECK(bound_expr);
  auto copied_expr = bound_expr->deep_copy();
  auto casted_expr = copied_expr->cast(ctx().fp64());
  CHECK(casted_expr);
  auto casted_constant = std::dynamic_pointer_cast<const Constant>(casted_expr);
  CHECK(casted_constant);
  return casted_constant->get_constval().doubleval;
}

int32_t WidthBucketExpr::get_partition_count_val() const {
  auto const_partition_count_expr = dynamic_cast<const Constant*>(partition_count_.get());
  if (!const_partition_count_expr) {
    return -1;
  }
  auto d = const_partition_count_expr->get_constval();
  switch (const_partition_count_expr->type()->id()) {
    case hdk::ir::Type::kInteger:
      switch (const_partition_count_expr->type()->size()) {
        case 1:
          return d.tinyintval;
        case 2:
          return d.smallintval;
        case 4:
          return d.intval;
        case 8: {
          auto bi = d.bigintval;
          if (bi < 1 || bi > INT32_MAX) {
            return -1;
          }
          return bi;
        }
        default:
          return -1;
      }
    default:
      return -1;
  }
}

bool expr_list_match(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(*lhs[i] == *rhs[i])) {
      return false;
    }
  }
  return true;
}

size_t Expr::hash() const {
  if (!hash_) {
    hash_ = typeid(*this).hash_code();
    boost::hash_combine(*hash_, type_->toString());
    boost::hash_combine(*hash_, contains_agg_);
  }
  return *hash_;
}

size_t ColumnRef::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, node_->toHash());
    boost::hash_combine(*hash_, idx_);
  }
  return *hash_;
}

size_t GroupColumnRef::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, idx_);
  }
  return *hash_;
}

size_t ColumnVar::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, rte_idx_);
    boost::hash_combine(*hash_, col_info_->hash());
  }
  return *hash_;
}

size_t ExpressionTuple::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    for (auto& expr : tuple_) {
      boost::hash_combine(*hash_, expr->hash());
    }
  }
  return *hash_;
}

size_t Var::hash() const {
  if (!hash_) {
    hash_ = ColumnVar::hash();
    boost::hash_combine(*hash_, which_row);
    boost::hash_combine(*hash_, varno);
  }
  return *hash_;
}

size_t Constant::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, is_null);
    if (!is_null) {
      if (type_->isArray()) {
        for (auto& expr : value_list) {
          boost::hash_combine(*hash_, expr->hash());
        }
      } else {
        boost::hash_combine(*hash_, ::hash(constval, type_));
      }
    }
  }
  return *hash_;
}

size_t UOper::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, optype);
    boost::hash_combine(*hash_, operand->hash());
  }
  return *hash_;
}

size_t BinOper::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, optype);
    boost::hash_combine(*hash_, qualifier);
    boost::hash_combine(*hash_, left_operand->hash());
    boost::hash_combine(*hash_, right_operand->hash());
  }
  return *hash_;
}

size_t RangeOper::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, left_inclusive_);
    boost::hash_combine(*hash_, right_inclusive_);
    boost::hash_combine(*hash_, left_operand_->hash());
    boost::hash_combine(*hash_, right_operand_->hash());
  }
  return *hash_;
}

size_t ScalarSubquery::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, node_->toHash());
  }
  return *hash_;
}

size_t InValues::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    for (auto& expr : value_list) {
      boost::hash_combine(*hash_, expr->hash());
    }
  }
  return *hash_;
}

size_t InIntegerSet::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    boost::hash_combine(*hash_, value_list);
  }
  return *hash_;
}

size_t InSubquery::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg_->hash());
    boost::hash_combine(*hash_, node_->toHash());
  }
  return *hash_;
}

size_t CharLengthExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    boost::hash_combine(*hash_, calc_encoded_length);
  }
  return *hash_;
}

size_t KeyForStringExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
  }
  return *hash_;
}

size_t SampleRatioExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
  }
  return *hash_;
}

size_t LowerExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
  }
  return *hash_;
}

size_t CardinalityExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
  }
  return *hash_;
}

size_t LikeExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    boost::hash_combine(*hash_, like_expr->hash());
    if (escape_expr) {
      boost::hash_combine(*hash_, escape_expr->hash());
    }
    boost::hash_combine(*hash_, is_ilike);
    boost::hash_combine(*hash_, is_simple);
  }
  return *hash_;
}

size_t RegexpExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    boost::hash_combine(*hash_, pattern_expr->hash());
    if (escape_expr) {
      boost::hash_combine(*hash_, escape_expr->hash());
    }
  }
  return *hash_;
}

size_t WidthBucketExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, target_value_->hash());
    boost::hash_combine(*hash_, lower_bound_->hash());
    boost::hash_combine(*hash_, upper_bound_->hash());
    boost::hash_combine(*hash_, partition_count_->hash());
    boost::hash_combine(*hash_, constant_expr_);
    boost::hash_combine(*hash_, skip_out_of_bound_check_);
  }
  return *hash_;
}

size_t LikelihoodExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, arg->hash());
    boost::hash_combine(*hash_, likelihood);
  }
  return *hash_;
}

size_t AggExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, aggtype);
    if (arg) {
      boost::hash_combine(*hash_, arg->hash());
    }
    boost::hash_combine(*hash_, is_distinct);
    if (arg1) {
      boost::hash_combine(*hash_, arg1->hash());
    }
  }
  return *hash_;
}

size_t CaseExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    for (auto& pr : expr_pair_list) {
      boost::hash_combine(*hash_, pr.first->hash());
      boost::hash_combine(*hash_, pr.second->hash());
    }
    if (else_expr) {
      boost::hash_combine(*hash_, else_expr->hash());
    }
  }
  return *hash_;
}

size_t ExtractExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, field_);
    boost::hash_combine(*hash_, from_expr_->hash());
  }
  return *hash_;
}

size_t DateaddExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, field_);
    boost::hash_combine(*hash_, number_->hash());
    boost::hash_combine(*hash_, datetime_->hash());
  }
  return *hash_;
}

size_t DatediffExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, field_);
    boost::hash_combine(*hash_, start_->hash());
    boost::hash_combine(*hash_, end_->hash());
  }
  return *hash_;
}

size_t DatetruncExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, field_);
    boost::hash_combine(*hash_, from_expr_->hash());
  }
  return *hash_;
}

size_t FunctionOper::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, name_);
    for (auto& expr : args_) {
      boost::hash_combine(*hash_, expr->hash());
    }
  }
  return *hash_;
}

size_t WindowFunction::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    boost::hash_combine(*hash_, kind_);
    for (auto& expr : args_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    for (auto& expr : partition_keys_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    for (auto& expr : order_keys_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    for (auto& collation : collation_) {
      boost::hash_combine(*hash_, collation.hash());
    }
  }
  return *hash_;
}

size_t ArrayExpr::hash() const {
  if (!hash_) {
    hash_ = Expr::hash();
    for (auto& expr : contained_expressions_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    boost::hash_combine(*hash_, local_alloc_);
    boost::hash_combine(*hash_, is_null_);
  }
  return *hash_;
}

size_t TargetEntry::hash() const {
  size_t res = 0;
  boost::hash_combine(res, resname);
  boost::hash_combine(res, expr->hash());
  boost::hash_combine(res, unnest);
  return res;
}

}  // namespace hdk::ir
