/*
 * Copyright 2020 OmniSci, Inc.
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

#include <cstddef>
#include "../Shared/sqltypes.h"
#include "IR/Type.h"
#include "Shared/types.h"

#include <functional>
#include <map>

#include "Logger/Logger.h"

struct ChunkStats {
  Datum min;
  Datum max;
  bool has_nulls;
};

template <typename T>
void fillChunkStats(ChunkStats& stats,
                    const hdk::ir::Type* type,
                    const T min,
                    const T max,
                    const bool has_nulls) {
  stats.has_nulls = has_nulls;
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (type->size()) {
        case 1:
          stats.min.tinyintval = min;
          stats.max.tinyintval = max;
          break;
        case 2:
          stats.min.smallintval = min;
          stats.max.smallintval = max;
          break;
        case 4:
          stats.min.intval = min;
          stats.max.intval = max;
          break;
        case 8:
          stats.min.bigintval = min;
          stats.max.bigintval = max;
          break;
        default:
          abort();
      }
      break;
    case hdk::ir::Type::kExtDictionary:
      stats.min.intval = min;
      stats.max.intval = max;
      break;
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kInterval:
      stats.min.bigintval = min;
      stats.max.bigintval = max;
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          stats.min.floatval = min;
          stats.max.floatval = max;
          break;
        case hdk::ir::FloatingPointType::kDouble:
          stats.min.doubleval = min;
          stats.max.doubleval = max;
          break;
        default:
          abort();
      }
      break;
    default:
      break;
  }
}

inline void mergeStats(ChunkStats& lhs,
                       const ChunkStats& rhs,
                       const hdk::ir::Type* type) {
  auto elem_type =
      type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
  lhs.has_nulls |= rhs.has_nulls;
  switch (elem_type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (elem_type->size()) {
        case 1:
          lhs.min.tinyintval = std::min(lhs.min.tinyintval, rhs.min.tinyintval);
          lhs.max.tinyintval = std::max(lhs.max.tinyintval, rhs.max.tinyintval);
          break;
        case 2:
          lhs.min.smallintval = std::min(lhs.min.smallintval, rhs.min.smallintval);
          lhs.max.smallintval = std::max(lhs.max.smallintval, rhs.max.smallintval);
          break;
        case 4:
          lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
          lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
          break;
        case 8:
          lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
          lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
          break;
        default:
          abort();
      }
      break;
    case hdk::ir::Type::kExtDictionary:
      lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
      lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
      break;
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kInterval:
      lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
      lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (elem_type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          lhs.min.floatval = std::min(lhs.min.floatval, rhs.min.floatval);
          lhs.max.floatval = std::max(lhs.max.floatval, rhs.max.floatval);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          lhs.min.doubleval = std::min(lhs.min.doubleval, rhs.min.doubleval);
          lhs.max.doubleval = std::max(lhs.max.doubleval, rhs.max.doubleval);
          break;
        default:
          abort();
      }
      break;
    default:
      break;
  }
}

class ChunkMetadata {
 public:
  using StatsMaterializeFn = std::function<void(ChunkStats&)>;

  ChunkMetadata(const hdk::ir::Type* type,
                const size_t num_bytes,
                const size_t num_elements)
      : type_(type), num_bytes_(num_bytes), num_elements_(num_elements) {}

  ChunkMetadata(const hdk::ir::Type* type,
                const size_t num_bytes,
                const size_t num_elements,
                const ChunkStats& chunk_stats)
      : type_(type)
      , num_bytes_(num_bytes)
      , num_elements_(num_elements)
      , chunk_stats_(chunk_stats) {}

  ChunkMetadata(const hdk::ir::Type* type,
                const size_t num_bytes,
                const size_t num_elements,
                StatsMaterializeFn stats_materialize_fn)
      : type_(type)
      , num_bytes_(num_bytes)
      , num_elements_(num_elements)
      , stats_materialize_fn_(std::move(stats_materialize_fn)) {}

  const hdk::ir::Type* type() const { return type_; }
  size_t numBytes() const { return num_bytes_; }
  size_t numElements() const { return num_elements_; }
  const ChunkStats& chunkStats() const {
    maybeMaterializeStats();
    return chunk_stats_;
  }

#ifndef __CUDACC__
  std::string dump() const {
    std::string res = "type: " + type_->toString() +
                      " numBytes: " + to_string(num_bytes_) + " numElements " +
                      to_string(num_elements_);
    auto elem_type =
        type_->isArray() ? type_->as<hdk::ir::ArrayBaseType>()->elemType() : type_;
    if (stats_materialize_fn_) {
      res +=
          " min: <not materialized> max: <not materialized> has_nulls: <not "
          "materialized>";
    } else if (elem_type->isString()) {
      // Unencoded strings have no min/max.
      res += " min: <invalid> max: <invalid> has_nulls: " +
             to_string(chunk_stats_.has_nulls);
    } else if (elem_type->isExtDictionary()) {
      res += " min: " + to_string(chunk_stats_.min.intval) +
             " max: " + to_string(chunk_stats_.max.intval) +
             " has_nulls: " + to_string(chunk_stats_.has_nulls);
    } else {
      res += " min: " + DatumToString(chunk_stats_.min, elem_type) +
             " max: " + DatumToString(chunk_stats_.max, elem_type) +
             " has_nulls: " + to_string(chunk_stats_.has_nulls);
    }

    return res;
  }

  std::string toString() const {
    return dump();
  }
#endif

  template <typename T>
  void fillChunkStats(const T min, const T max, const bool has_nulls) {
    StatsMaterializeFn().swap(stats_materialize_fn_);
    ::fillChunkStats(chunk_stats_, type_, min, max, has_nulls);
  }

  void fillChunkStats(const ChunkStats& new_stats) {
    StatsMaterializeFn().swap(stats_materialize_fn_);
    chunk_stats_ = new_stats;
  }

  void fillChunkStats(const Datum min, const Datum max, const bool has_nulls) {
    StatsMaterializeFn().swap(stats_materialize_fn_);
    chunk_stats_.has_nulls = has_nulls;
    chunk_stats_.min = min;
    chunk_stats_.max = max;
  }

  void fillStringChunkStats(const bool has_nulls) {
    StatsMaterializeFn().swap(stats_materialize_fn_);
    chunk_stats_.has_nulls = has_nulls;
#ifndef __CUDACC__
    chunk_stats_.min.stringval = nullptr;
    chunk_stats_.max.stringval = nullptr;
#endif
  }

  bool operator==(const ChunkMetadata& that) const {
    if (!type_->equal(that.type_) || num_bytes_ != that.num_bytes_ ||
        num_elements_ != that.num_elements_) {
      return false;
    }

    maybeMaterializeStats();
    that.maybeMaterializeStats();

    return DatumEqual(chunk_stats_.min,
                      that.chunk_stats_.min,
                      type_->isArray() ? type_->as<hdk::ir::ArrayBaseType>()->elemType()
                                       : type_) &&
           DatumEqual(chunk_stats_.max,
                      that.chunk_stats_.max,
                      type_->isArray() ? type_->as<hdk::ir::ArrayBaseType>()->elemType()
                                       : type_) &&
           chunk_stats_.has_nulls == that.chunk_stats_.has_nulls;
  }

 private:
  void maybeMaterializeStats() const {
    if (stats_materialize_fn_) {
      stats_materialize_fn_(chunk_stats_);
      StatsMaterializeFn().swap(stats_materialize_fn_);
    }
  }

  const hdk::ir::Type* type_;
  size_t num_bytes_;
  size_t num_elements_;
  mutable ChunkStats chunk_stats_;
  mutable StatsMaterializeFn stats_materialize_fn_;
};

inline int64_t extract_min_stat_int_type(const ChunkStats& stats,
                                         const hdk::ir::Type* type) {
  return extract_int_type_from_datum(stats.min, type);
}

inline int64_t extract_max_stat_int_type(const ChunkStats& stats,
                                         const hdk::ir::Type* type) {
  return extract_int_type_from_datum(stats.max, type);
}

inline double extract_min_stat_fp_type(const ChunkStats& stats,
                                       const hdk::ir::Type* type) {
  return extract_fp_type_from_datum(stats.min, type);
}

inline double extract_max_stat_fp_type(const ChunkStats& stats,
                                       const hdk::ir::Type* type) {
  return extract_fp_type_from_datum(stats.max, type);
}

using ChunkMetadataMap = std::map<int, std::shared_ptr<ChunkMetadata>>;
using ChunkMetadataVector =
    std::vector<std::pair<ChunkKey, std::shared_ptr<ChunkMetadata>>>;
