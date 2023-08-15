/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2023 Intel Corporation
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
  kBwAnd,
  kBwOr,
  kBwXor,
  kBwNot,
  kMinus,
  kPlus,
  kMul,
  kDiv,
  kMod,
  kUMinus,
  kIsNull,
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
         op == OpType::kCast || op == OpType::kBwNot;
}
inline bool isEquivalence(OpType op) {
  return op == OpType::kEq || op == OpType::kBwEq;
}

inline bool isBitwise(OpType op) {
  return op == OpType::kBwAnd || op == OpType::kBwOr || op == OpType::kBwXor ||
         op == OpType::kBwNot;
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
  kSingleValue,
  kTopK,
  // Compound aggregates
  kStdDevSamp,
  kCorr,
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
