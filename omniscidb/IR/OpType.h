/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace hdk::ir {

// must not change the order without keeping the array in OperExpr::to_string
// in sync.
enum class OpType {
  kEq = 0,
  kBwEq,
  kNe,
  kLt,
  kGt,
  kLe,
  kGe,
  kAnd,
  kOr,
  kNot,
  kMinus,
  kPlus,
  kMul,
  kDiv,
  kMod,
  kUMinus,
  kIsNull,
  kIsNotNull,
  kExists,
  kCast,
  kArrayAt,
  kUnnest,
  kFunction,
  kIn
};

inline bool isComparison(OpType op) {
  return op == OpType::kEq || op == OpType::kBwEq || op == OpType::kNe ||
         op == OpType::kLt || op == OpType::kGt || op == OpType::kLe || op == OpType::kGe;
}

inline bool isLogic(OpType op) {
  return op == OpType::kAnd || op == OpType::kOr;
}

inline bool isArithmetic(OpType op) {
  return op == OpType::kMinus || op == OpType::kPlus || op == OpType::kMul ||
         op == OpType::kDiv || op == OpType::kMod;
}
inline OpType commuteComparison(OpType op) {
  return op == OpType::kLt   ? OpType::kGt
         : op == OpType::kLe ? OpType::kGe
         : op == OpType::kGt ? OpType::kLt
         : op == OpType::kGe ? OpType::kLe
                             : op;
}
inline bool isUnary(OpType op) {
  return op == OpType::kNot || op == OpType::kUMinus || op == OpType::kIsNull ||
         op == OpType::kExists || op == OpType::kCast;
}
inline bool isEquivalence(OpType op) {
  return op == OpType::kEq || op == OpType::kBwEq;
}

enum class Qualifier { kOne, kAny, kAll };

enum class AggType {
  kAvg,
  kMin,
  kMax,
  kSum,
  kCount,
  kApproxCountDistinct,
  kApproxQuantile,
  kSample,
  kSingleValue
};

enum class WindowFunctionKind {
  RowNumber,
  Rank,
  DenseRank,
  PercentRank,
  CumeDist,
  NTile,
  Lag,
  Lead,
  FirstValue,
  LastValue,
  Avg,
  Min,
  Max,
  Sum,
  Count,
  SumInternal  // For deserialization from Calcite only. Gets rewritten to a regular SUM.
};

}  // namespace hdk::ir

inline std::string toString(hdk::ir::OpType op) {
  switch (op) {
    case hdk::ir::OpType::kEq:
      return "EQ";
    case hdk::ir::OpType::kBwEq:
      return "BW_EQ";
    case hdk::ir::OpType::kNe:
      return "NE";
    case hdk::ir::OpType::kLt:
      return "LT";
    case hdk::ir::OpType::kGt:
      return "GT";
    case hdk::ir::OpType::kLe:
      return "LE";
    case hdk::ir::OpType::kGe:
      return "GE";
    case hdk::ir::OpType::kAnd:
      return "AND";
    case hdk::ir::OpType::kOr:
      return "OR";
    case hdk::ir::OpType::kNot:
      return "NOT";
    case hdk::ir::OpType::kMinus:
      return "MINUS";
    case hdk::ir::OpType::kPlus:
      return "PLUS";
    case hdk::ir::OpType::kMul:
      return "MULTIPLY";
    case hdk::ir::OpType::kDiv:
      return "DIVIDE";
    case hdk::ir::OpType::kMod:
      return "MODULO";
    case hdk::ir::OpType::kUMinus:
      return "UMINUS";
    case hdk::ir::OpType::kIsNull:
      return "ISNULL";
    case hdk::ir::OpType::kIsNotNull:
      return "ISNOTNULL";
    case hdk::ir::OpType::kExists:
      return "EXISTS";
    case hdk::ir::OpType::kCast:
      return "CAST";
    case hdk::ir::OpType::kArrayAt:
      return "ARRAY_AT";
    case hdk::ir::OpType::kUnnest:
      return "UNNEST";
    case hdk::ir::OpType::kFunction:
      return "FUNCTION";
    case hdk::ir::OpType::kIn:
      return "IN";
  }
  LOG(FATAL) << "Invalid operation kind: " << (int)op;
  return "";
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::OpType op) {
  return os << toString(op);
}

inline std::string toString(hdk::ir::Qualifier qualifier) {
  switch (qualifier) {
    case hdk::ir::Qualifier::kOne:
      return "ONE";
    case hdk::ir::Qualifier::kAny:
      return "ANY";
    case hdk::ir::Qualifier::kAll:
      return "ALL";
  }
  LOG(FATAL) << "Invalid Qualifier: " << int(qualifier);
  return "";
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::Qualifier qualifier) {
  return os << toString(qualifier);
}

inline std::string toString(hdk::ir::AggType agg) {
  switch (agg) {
    case hdk::ir::AggType::kAvg:
      return "AVG";
    case hdk::ir::AggType::kMin:
      return "MIN";
    case hdk::ir::AggType::kMax:
      return "MAX";
    case hdk::ir::AggType::kSum:
      return "SUM";
    case hdk::ir::AggType::kCount:
      return "COUNT";
    case hdk::ir::AggType::kApproxCountDistinct:
      return "APPROX_COUNT_DISTINCT";
    case hdk::ir::AggType::kApproxQuantile:
      return "APPROX_PERCENTILE";
    case hdk::ir::AggType::kSample:
      return "SAMPLE";
    case hdk::ir::AggType::kSingleValue:
      return "SINGLE_VALUE";
  }
  LOG(FATAL) << "Invalid aggregate kind: " << (int)agg;
  return "";
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::AggType agg) {
  return os << toString(agg);
}

inline std::string toString(hdk::ir::WindowFunctionKind kind) {
  switch (kind) {
    case hdk::ir::WindowFunctionKind::RowNumber:
      return "ROW_NUMBER";
    case hdk::ir::WindowFunctionKind::Rank:
      return "RANK";
    case hdk::ir::WindowFunctionKind::DenseRank:
      return "DENSE_RANK";
    case hdk::ir::WindowFunctionKind::PercentRank:
      return "PERCENT_RANK";
    case hdk::ir::WindowFunctionKind::CumeDist:
      return "CUME_DIST";
    case hdk::ir::WindowFunctionKind::NTile:
      return "NTILE";
    case hdk::ir::WindowFunctionKind::Lag:
      return "LAG";
    case hdk::ir::WindowFunctionKind::Lead:
      return "LEAD";
    case hdk::ir::WindowFunctionKind::FirstValue:
      return "FIRST_VALUE";
    case hdk::ir::WindowFunctionKind::LastValue:
      return "LAST_VALUE";
    case hdk::ir::WindowFunctionKind::Avg:
      return "AVG";
    case hdk::ir::WindowFunctionKind::Min:
      return "MIN";
    case hdk::ir::WindowFunctionKind::Max:
      return "MAX";
    case hdk::ir::WindowFunctionKind::Sum:
      return "SUM";
    case hdk::ir::WindowFunctionKind::Count:
      return "COUNT";
    case hdk::ir::WindowFunctionKind::SumInternal:
      return "SUM_INTERNAL";
  }
  LOG(FATAL) << "Invalid window function kind.";
  return "";
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::WindowFunctionKind kind) {
  return os << toString(kind);
}
