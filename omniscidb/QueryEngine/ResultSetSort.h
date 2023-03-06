/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSet.h"

#include <functional>

using Comparator = std::function<bool(const PermutationIdx, const PermutationIdx)>;
using ApproxQuantileBuffers = std::vector<std::vector<double>>;

template <typename BUFFER_ITERATOR_TYPE>
struct ResultSetComparator {
  using BufferIteratorType = BUFFER_ITERATOR_TYPE;

  ResultSetComparator(const std::list<hdk::ir::OrderEntry>& order_entries,
                      const ResultSet* result_set,
                      const PermutationView permutation,
                      const Executor* executor,
                      const bool single_threaded)
      : order_entries_(order_entries)
      , result_set_(result_set)
      , permutation_(permutation)
      , buffer_itr_(result_set)
      , executor_(executor)
      , single_threaded_(single_threaded)
      , approx_quantile_materialized_buffers_(materializeApproxQuantileColumns()) {
    materializeCountDistinctColumns();
  }

  void materializeCountDistinctColumns();
  ApproxQuantileBuffers materializeApproxQuantileColumns() const;

  std::vector<int64_t> materializeCountDistinctColumn(
      const hdk::ir::OrderEntry& order_entry) const;
  ApproxQuantileBuffers::value_type materializeApproxQuantileColumn(
      const hdk::ir::OrderEntry& order_entry) const;

  bool operator()(const PermutationIdx lhs, const PermutationIdx rhs) const;

  const std::list<hdk::ir::OrderEntry>& order_entries_;
  const ResultSet* result_set_;
  const PermutationView permutation_;
  const BufferIteratorType buffer_itr_;
  const Executor* executor_;
  const bool single_threaded_;
  std::vector<std::vector<int64_t>> count_distinct_materialized_buffers_;
  const ApproxQuantileBuffers approx_quantile_materialized_buffers_;
};

template struct ResultSetComparator<ResultSet::RowWiseTargetAccessor>;
template struct ResultSetComparator<ResultSet::ColumnWiseTargetAccessor>;

Comparator createComparator(ResultSet* rs,
                            const std::list<hdk::ir::OrderEntry>& order_entries,
                            const PermutationView permutation,
                            const Executor* executor,
                            const bool single_threaded);

PermutationView topPermutation(PermutationView permutation,
                               const size_t n,
                               const Comparator&,
                               const bool single_threaded);

void sortResultSet(ResultSet* rs,
                   const std::list<hdk::ir::OrderEntry>& order_entries,
                   size_t top_n,
                   const Executor* executor);
