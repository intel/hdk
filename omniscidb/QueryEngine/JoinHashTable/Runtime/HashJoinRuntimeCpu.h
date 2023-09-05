/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "Logger/Logger.h"
#include "StringDictionary/StringDictionary.h"
#include "StringDictionary/StringDictionaryProxy.h"

#include <cstddef>
#include <cstdint>

/**
 * Joins between two dictionary encoded string columns without a shared string dictionary
 * are computed by translating the inner dictionary to the outer dictionary while filling
 * the  hash table. The translation works as follows:
 *
 * Given two tables t1 and t2, with t1 the outer table and t2 the inner table, and two
 * columns t1.x and t2.x, both dictionary encoded strings without a shared dictionary, we
 * read each value in t2.x and do a lookup in the dictionary for t1.x. If the lookup
 * returns a valid ID, we insert that ID into the hash table. Otherwise, we skip adding an
 * entry into the hash table for the inner column. We can also skip adding any entries
 * that are outside the range of the outer column.
 *
 * Consider a join of the form SELECT x, n FROM (SELECT x, COUNT(*) n FROM t1 GROUP BY x
 * HAVING n > 10), t2 WHERE t1.x = t2.x; Let the result of the subquery be t1_s.
 * Due to the HAVING clause, the range of all IDs in t1_s must be less than or equal to
 * the range of all IDs in t1. Suppose we have an element a in t2.x that is also in
 * t1_s.x. Then the ID of a must be within the range of t1_s. Therefore it is safe to
 * ignore any element ID that is not in the dictionary corresponding to t1_s.x or is
 * outside the range of column t1_s.
 */

inline int64_t translate_str_id_to_outer_dict(const int64_t elem,
                                              const int64_t min_elem,
                                              const int64_t max_elem,
                                              const void* sd_inner_proxy,
                                              const void* sd_outer_proxy) {
  CHECK(sd_outer_proxy);
  const auto sd_inner_dict_proxy =
      static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
  const auto sd_outer_dict_proxy =
      static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
  const auto elem_str = sd_inner_dict_proxy->getString(elem);
  const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
  if (outer_id > max_elem || outer_id < min_elem) {
    return StringDictionary::INVALID_STR_ID;
  }
  return outer_id;
};

inline int64_t map_str_id_to_outer_dict(const int64_t inner_elem,
                                        const int64_t min_outer_elem,
                                        const int64_t max_outer_elem,
                                        const int32_t* inner_to_outer_translation_map) {
  const auto outer_id = inner_to_outer_translation_map[inner_elem];
  if (outer_id > max_outer_elem || outer_id < min_outer_elem) {
    return StringDictionary::INVALID_STR_ID;
  }
  return outer_id;
}

struct JoinColumn;
struct JoinColumnTypeInfo;

int fill_hash_join_buff_bucketized_cpu(int32_t* cpu_hash_table_buff,
                                       const int32_t hash_join_invalid_val,
                                       const bool for_semi_join,
                                       const JoinColumn& join_column,
                                       const JoinColumnTypeInfo& type_info,
                                       const int32_t* sd_inner_to_outer_translation_map,
                                       const int64_t bucket_normalization);
