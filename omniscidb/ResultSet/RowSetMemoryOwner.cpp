/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2021 OmniSci, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RowSetMemoryOwner.h"

#include "Shared/approx_quantile.h"
#include "Shared/quantile.h"

EXTERN extern bool g_cache_string_hash;
EXTERN extern size_t g_approx_quantile_buffer;
EXTERN extern size_t g_approx_quantile_centroids;

RowSetMemoryOwner::~RowSetMemoryOwner() {
  for (auto count_distinct_set : count_distinct_sets_) {
    delete count_distinct_set;
  }
  for (auto group_by_buffer : group_by_buffers_) {
    free(group_by_buffer);
  }
  for (auto varlen_buffer : varlen_buffers_) {
    free(varlen_buffer);
  }
  for (auto varlen_input_buffer : varlen_input_buffers_) {
    CHECK(varlen_input_buffer);
    varlen_input_buffer->unPin();
  }
  for (auto col_buffer : col_buffers_) {
    free(col_buffer);
  }
  for (auto& pr : quantiles_) {
    for (size_t i = 0; i < pr.second; ++i) {
      pr.first[i].~Quantile();
    }
  }
}

StringDictionary* RowSetMemoryOwner::getOrAddStringDictProxy(const int dict_id_in,
                                                             const int64_t generation) {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(data_provider_);
  const auto dd = data_provider_->getDictMetadata(dict_id);
  if (dd) {
    CHECK(dd->stringDict);
    CHECK_LE(dd->dictNBits, 32);
    return addStringDict(dd->stringDict, dict_id, generation);
  }
  // It's possible the original dictionary has been removed from its storage
  // but we still have it in a proxy.
  if (dict_id != DictRef::literalsDictId) {
    auto res = getStringDictProxyOwned(dict_id_in);
    CHECK(res) << "Cannot find dict or proxy " << dict_id_in;
    CHECK(generation < 0 || res->getBaseGeneration() == generation);
    return res.get();
  }
  CHECK_EQ(dict_id, DictRef::literalsDictId);
  if (!lit_str_dict_proxy_) {
    DictRef literal_dict_ref(DictRef::invalidDbId, DictRef::literalsDictId);
    std::shared_ptr<StringDictionary> tsd =
        std::make_shared<StringDictionary>(literal_dict_ref, g_cache_string_hash);
    lit_str_dict_proxy_ = std::make_shared<StringDictionary>(tsd, 0);
  }
  return lit_str_dict_proxy_.get();
}

std::shared_ptr<StringDictionary> RowSetMemoryOwner::getStringDictProxyOwned(
    const int dict_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (dict_id == DictRef::literalsDictId) {
    return lit_str_dict_proxy_;
  }
  auto it = str_dict_proxy_owned_.find(dict_id);
  if (it != str_dict_proxy_owned_.end()) {
    return it->second;
  }
  return nullptr;
}

quantile::TDigest* RowSetMemoryOwner::nullTDigest(double const q) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return t_digests_
      .emplace_back(std::make_unique<quantile::TDigest>(
          q, this, g_approx_quantile_buffer, g_approx_quantile_centroids))
      .get();
}

hdk::quantile::Quantile* RowSetMemoryOwner::quantiles(size_t count) {
  hdk::quantile::Quantile* quantile_arr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    quantile_arr = reinterpret_cast<hdk::quantile::Quantile*>(
        allocateNoLock(count * sizeof(hdk::quantile::Quantile)));
    quantiles_.emplace_back(quantile_arr, count);
  }
  for (size_t i = 0; i < count; ++i) {
    new (quantile_arr + i) hdk::quantile::Quantile(this);
  }
  return quantile_arr;
}

int8_t* RowSetMemoryOwner::topKBuffer(size_t size) {
  return allocate(size);
}
