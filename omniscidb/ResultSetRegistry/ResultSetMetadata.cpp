/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ResultSetMetadata.h"

#include "Shared/thread_count.h"

#include <future>

namespace hdk {

namespace {

bool usesIntMeta(const hdk::ir::Type* col_type) {
  return col_type->isInteger() || col_type->isDecimal() || col_type->isDateTime() ||
         col_type->isBoolean() || col_type->isExtDictionary();
}

void updateStats(const ScalarTargetValue* scalar_col_val,
                 Encoder* encoder,
                 const hdk::ir::Type* col_type,
                 bool skip_nulls) {
  if (usesIntMeta(col_type)) {
    const auto i64_p = boost::get<int64_t>(scalar_col_val);
    CHECK(i64_p);
    if (!skip_nulls || *i64_p != inline_int_null_value(col_type)) {
      encoder->updateStats(*i64_p, *i64_p == inline_int_null_value(col_type));
    }
  } else if (col_type->isFloatingPoint()) {
    switch (col_type->as<hdk::ir::FloatingPointType>()->precision()) {
      case hdk::ir::FloatingPointType::kFloat: {
        const auto float_p = boost::get<float>(scalar_col_val);
        CHECK(float_p);
        if (!skip_nulls || *float_p != inline_fp_null_value(col_type)) {
          encoder->updateStats(*float_p, *float_p == inline_fp_null_value(col_type));
        }
        break;
      }
      case hdk::ir::FloatingPointType::kDouble: {
        const auto double_p = boost::get<double>(scalar_col_val);
        CHECK(double_p);
        if (!skip_nulls || *double_p != inline_fp_null_value(col_type)) {
          encoder->updateStats(*double_p, *double_p == inline_fp_null_value(col_type));
        }
        break;
      }
      default:
        CHECK(false);
    }
  } else if (col_type->isString()) {
    const auto str_p = boost::get<NullableString>(scalar_col_val);
    CHECK(str_p);
    // updateStats API for StringNoneEncoder makes us to make a copy of the string
    // just to check if its empty and it doesn't differ empty strings from NULLs.
    // So use resetChunkStats instead because it simply copies has_null value from
    // passed stats which exactly what we need here.
    if (str_p->type() == typeid(void*) && !skip_nulls) {
      encoder->resetChunkStats({{}, {}, true});
    }
  } else {
    throw std::runtime_error(col_type->toString() +
                             " is not supported in temporary table.");
  }
}

}  // namespace

ChunkMetadataMap synthesizeMetadata(const ResultSet* rows) {
  auto timer = DEBUG_TIMER(__func__);

  ChunkMetadataMap metadata_map;

  if (rows->definitelyHasNoRows()) {
    // resultset has no valid storage, so we fill dummy metadata and return early
    std::vector<std::unique_ptr<Encoder>> decoders;
    for (size_t i = 0; i < rows->colCount(); ++i) {
      decoders.emplace_back(Encoder::Create(nullptr, rows->colType(i)));
      const auto it_ok =
          metadata_map.emplace(i + 1, decoders.back()->getMetadata(rows->colType(i)));
      CHECK(it_ok.second);
    }
    return metadata_map;
  }

  std::vector<std::vector<std::unique_ptr<Encoder>>> dummy_encoders;
  std::vector<std::vector<size_t>> varlen_lengths;
  const size_t worker_count =
      result_set::use_parallel_algorithms(*rows) ? cpu_threads() : 1;
  bool has_varlen = false;
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    dummy_encoders.emplace_back();
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_type = rows->colType(i);
      if (col_type->isArray()) {
        // For arrays we create encoders for elem type to avoid creating ArrayDatum to
        // update stats. This requires special handling for nulls when we update stats.
        dummy_encoders.back().emplace_back(
            Encoder::Create(nullptr, col_type->as<hdk::ir::ArrayBaseType>()->elemType()));
      } else {
        dummy_encoders.back().emplace_back(Encoder::Create(nullptr, col_type));
      }
      has_varlen = has_varlen || col_type->isVarLen();
    }
  }
  if (has_varlen) {
    varlen_lengths.resize(worker_count, std::vector<size_t>(rows->colCount(), 0));
  }

  rows->moveToBegin();
  const auto do_work = [rows](const std::vector<TargetValue>& crt_row,
                              std::vector<std::unique_ptr<Encoder>>& dummy_encoders,
                              std::vector<size_t>& varlen_lengths) {
    for (size_t i = 0; i < rows->colCount(); ++i) {
      auto col_type = rows->colType(i);
      const auto& col_val = crt_row[i];

      if (auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val)) {
        updateStats(scalar_col_val, dummy_encoders[i].get(), col_type, false);
        if (auto nullable_str_p = boost::get<NullableString>(scalar_col_val)) {
          if (auto str_p = boost::get<std::string>(nullable_str_p)) {
            varlen_lengths[i] += str_p->size();
          }
        }
      } else {
        auto arr_col_val = boost::get<ArrayTargetValue>(&col_val);
        CHECK(arr_col_val);
        CHECK(col_type->isArray());
        if (*arr_col_val) {
          auto elem_type = col_type->as<hdk::ir::ArrayBaseType>()->elemType();
          for (auto& elem_val : **arr_col_val) {
            updateStats(&elem_val, dummy_encoders[i].get(), elem_type, true);
          }
          if (col_type->isVarLenArray()) {
            varlen_lengths[i] += (*arr_col_val)->size() * elem_type->size();
          } else {
            // Emtpy value provided for fixed length array type is considered as NULL.
            CHECK(col_type->isFixedLenArray());
            if ((*arr_col_val)->empty()) {
              dummy_encoders[i]->updateStats((int64_t)0, true);
            }
          }
        } else {
          dummy_encoders[i]->updateStats((int64_t)0, true);
        }
      }
    }
  };
  if (result_set::use_parallel_algorithms(*rows)) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> compute_stats_threads;
    const auto entry_count = rows->entryCount();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      compute_stats_threads.push_back(std::async(
          std::launch::async,
          [rows, &do_work, &dummy_encoders, &varlen_lengths](
              const size_t start, const size_t end, const size_t worker_idx) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows->getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(crt_row, dummy_encoders[worker_idx], varlen_lengths[worker_idx]);
              }
            }
          },
          start_entry,
          end_entry,
          i));
    }
    for (auto& child : compute_stats_threads) {
      child.wait();
    }
    for (auto& child : compute_stats_threads) {
      child.get();
    }
  } else {
    while (true) {
      auto crt_row = rows->getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(crt_row, dummy_encoders[0], varlen_lengths[0]);
    }
  }
  rows->moveToBegin();
  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    CHECK_LT(worker_idx, dummy_encoders.size());
    const auto& worker_encoders = dummy_encoders[worker_idx];
    for (size_t i = 0; i < rows->colCount(); ++i) {
      dummy_encoders[0][i]->reduceStats(*worker_encoders[i]);
      if (rows->colType(i)->isVarLenArray()) {
        varlen_lengths[0][i] += varlen_lengths[worker_idx][i];
      }
    }
  }
  for (size_t i = 0; i < rows->colCount(); ++i) {
    auto col_type = rows->colType(i);
    auto elem_type = col_type->isVarLenArray()
                         ? col_type->as<hdk::ir::ArrayBaseType>()->elemType()
                         : col_type;
    size_t num_bytes =
        col_type->isVarLen() ? varlen_lengths[0][i] : rows->rowCount() * col_type->size();
    auto meta = std::make_shared<ChunkMetadata>(
        col_type,
        num_bytes,
        rows->rowCount(),
        dummy_encoders[0][i]->getMetadata(elem_type)->chunkStats());
    const auto it_ok = metadata_map.emplace(i + 1, meta);
    CHECK(it_ok.second);
  }
  return metadata_map;
}

}  // namespace hdk
