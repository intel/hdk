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

StringDictionaryProxy* RowSetMemoryOwner::getOrAddStringDictProxy(
    const int dict_id_in,
    const int64_t generation) {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(data_provider_);
  const auto dd = data_provider_->getDictMetadata(dict_id);
  if (dd) {
    CHECK(dd->stringDict);
    CHECK_LE(dd->dictNBits, 32);
    return addStringDict(dd->stringDict, dict_id, generation);
  }
  CHECK_EQ(dict_id, DictRef::literalsDictId);
  if (!lit_str_dict_proxy_) {
    DictRef literal_dict_ref(DictRef::invalidDbId, DictRef::literalsDictId);
    std::shared_ptr<StringDictionary> tsd =
        std::make_shared<StringDictionary>(literal_dict_ref, g_cache_string_hash);
    lit_str_dict_proxy_ =
        std::make_shared<StringDictionaryProxy>(tsd, literal_dict_ref.dictId, 0);
  }
  return lit_str_dict_proxy_.get();
}

quantile::TDigest* RowSetMemoryOwner::nullTDigest(double const q) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return t_digests_
      .emplace_back(std::make_unique<quantile::TDigest>(
          q, this, g_approx_quantile_buffer, g_approx_quantile_centroids))
      .get();
}

hdk::quantile::Quantile* RowSetMemoryOwner::quantile() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return quantiles_.emplace_back(std::make_unique<hdk::quantile::Quantile>(this)).get();
}

int8_t* RowSetMemoryOwner::topKBuffer(size_t size) {
  return allocate(size);
}
