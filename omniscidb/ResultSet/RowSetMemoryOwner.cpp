/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2021 OmniSci, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RowSetMemoryOwner.h"

EXTERN extern bool g_cache_string_hash;

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
    std::shared_ptr<StringDictionary> tsd = std::make_shared<StringDictionary>(
        literal_dict_ref, "", false, true, g_cache_string_hash);
    lit_str_dict_proxy_ =
        std::make_shared<StringDictionaryProxy>(tsd, literal_dict_ref.dictId, 0);
  }
  return lit_str_dict_proxy_.get();
}
