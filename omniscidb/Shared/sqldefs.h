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

/**
 * @file		sqldefs.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Common Enum definitions for SQL processing.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLDEFS_H
#define SQLDEFS_H

enum SQLQualifier { kONE, kANY, kALL };

enum SQLAgg {
  kAVG,
  kMIN,
  kMAX,
  kSUM,
  kCOUNT,
  kAPPROX_COUNT_DISTINCT,
  kAPPROX_QUANTILE,
  kSAMPLE,
  kSINGLE_VALUE
};

enum class SqlWindowFunctionKind {
  ROW_NUMBER,
  RANK,
  DENSE_RANK,
  PERCENT_RANK,
  CUME_DIST,
  NTILE,
  LAG,
  LEAD,
  FIRST_VALUE,
  LAST_VALUE,
  AVG,
  MIN,
  MAX,
  SUM,
  COUNT,
  SUM_INTERNAL  // For deserialization from Calcite only. Gets rewritten to a regular SUM.
};

enum SQLStmtType { kSELECT, kUPDATE, kINSERT, kDELETE, kCREATE_TABLE };

enum StorageOption { kDISK = 0, kGPU = 1, kCPU = 2 };

enum ViewRefreshOption { kMANUAL = 0, kAUTO = 1, kIMMEDIATE = 2 };

enum class JoinType { INNER, LEFT, SEMI, ANTI, INVALID };

#ifndef __CUDACC__

#include <string>
#include "Logger/Logger.h"

inline std::string toString(const JoinType& join_type) {
  switch (join_type) {
    case JoinType::INNER:
      return "INNER";
    case JoinType::LEFT:
      return "LEFT";
    case JoinType::SEMI:
      return "SEMI";
    case JoinType::ANTI:
      return "ANTI";
    default:
      return "INVALID";
  }
}

inline std::string toString(const SQLQualifier& qualifier) {
  switch (qualifier) {
    case kONE:
      return "ONE";
    case kANY:
      return "ANY";
    case kALL:
      return "ALL";
  }
  LOG(FATAL) << "Invalid SQLQualifier: " << qualifier;
  return "";
}

inline std::string toString(const SQLAgg& kind) {
  switch (kind) {
    case kAVG:
      return "AVG";
    case kMIN:
      return "MIN";
    case kMAX:
      return "MAX";
    case kSUM:
      return "SUM";
    case kCOUNT:
      return "COUNT";
    case kAPPROX_COUNT_DISTINCT:
      return "APPROX_COUNT_DISTINCT";
    case kAPPROX_QUANTILE:
      return "APPROX_PERCENTILE";
    case kSAMPLE:
      return "SAMPLE";
    case kSINGLE_VALUE:
      return "SINGLE_VALUE";
  }
  LOG(FATAL) << "Invalid aggregate kind: " << kind;
  return "";
}

inline std::string toString(const SqlWindowFunctionKind& kind) {
  switch (kind) {
    case SqlWindowFunctionKind::ROW_NUMBER:
      return "ROW_NUMBER";
    case SqlWindowFunctionKind::RANK:
      return "RANK";
    case SqlWindowFunctionKind::DENSE_RANK:
      return "DENSE_RANK";
    case SqlWindowFunctionKind::PERCENT_RANK:
      return "PERCENT_RANK";
    case SqlWindowFunctionKind::CUME_DIST:
      return "CUME_DIST";
    case SqlWindowFunctionKind::NTILE:
      return "NTILE";
    case SqlWindowFunctionKind::LAG:
      return "LAG";
    case SqlWindowFunctionKind::LEAD:
      return "LEAD";
    case SqlWindowFunctionKind::FIRST_VALUE:
      return "FIRST_VALUE";
    case SqlWindowFunctionKind::LAST_VALUE:
      return "LAST_VALUE";
    case SqlWindowFunctionKind::AVG:
      return "AVG";
    case SqlWindowFunctionKind::MIN:
      return "MIN";
    case SqlWindowFunctionKind::MAX:
      return "MAX";
    case SqlWindowFunctionKind::SUM:
      return "SUM";
    case SqlWindowFunctionKind::COUNT:
      return "COUNT";
    case SqlWindowFunctionKind::SUM_INTERNAL:
      return "SUM_INTERNAL";
  }
  LOG(FATAL) << "Invalid window function kind.";
  return "";
}

#endif

#endif  // SQLDEFS_H
