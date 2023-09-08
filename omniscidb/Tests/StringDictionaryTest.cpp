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

#include "TestHelpers.h"

#include "Shared/scope.h"
#include "StringDictionary/StringDictionaryProxy.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

using namespace std::string_literals;

EXTERN extern bool g_cache_string_hash;
EXTERN extern bool g_enable_stringdict_parallel;

TEST(StringDictionary, AddAndGet) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("foo bar");
  auto id2 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(0, id1);
  auto id3 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id3);
  auto id4 = string_dict.getIdOfString(std::string_view("foo bar"));
  ASSERT_EQ(id1, id4);
  ASSERT_EQ("foo bar", string_dict.getString(id4));
  auto id5 = string_dict.getIdOfString(std::string("foo bar"));
  ASSERT_EQ(id1, id5);
  ASSERT_EQ("foo bar", string_dict.getString(id5));
}

TEST(StringDictionary, HandleEmpty) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("");
  auto id2 = string_dict.getOrAdd("");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), id1);
}

TEST(StringDictionary, RecoverZero) {
  const DictRef dict_ref(-1, 1);
  {
    StringDictionary string_dict(dict_ref, g_cache_string_hash);
    size_t num_strings = string_dict.storageEntryCount();
    ASSERT_EQ(static_cast<size_t>(0), num_strings);
  }
  StringDictionary string_dict(dict_ref, g_cache_string_hash);
  size_t num_strings = string_dict.storageEntryCount();
  ASSERT_EQ(static_cast<size_t>(0), num_strings);
}

const int g_op_count{250000};

TEST(StringDictionary, ManyAddsAndGets) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, g_cache_string_hash);
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(std::to_string(i), string_dict.getString(i));
  }
}

TEST(StringDictionary, GetOrAddBulk) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, g_cache_string_hash);
  {
    // First insert all new strings
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }
    string_dict.getOrAddBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now insert half new, half existing strings
    // Even idxs are existing, odd are new
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count + i / 2));
      }
    }
    string_dict.getOrAddBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      // Even though the if and else branches result in the same test, both are
      // left help diagnose the source of test failures
      // (existing strings for the if branch and new strings for the else branch)
      if (i % 2 == 0) {
        // Existing strings
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        // New strings
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      }
    }
  }

  // test GetBulk
  {
    // First iteration is identical to first of GetOrAddBulk, and results should be the
    // same
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }
    string_dict.getBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }
  {
    // Second iteration - even idxs exist, odd do not and should return INVALID_STR_ID
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        // Won't exist in dictionary
        strings.emplace_back(std::to_string(g_op_count * 2 + i));
      }
    }
    string_dict.getBulk(strings, string_ids.data());
    const std::string invalid_str_id_as_string =
        std::to_string(StringDictionary::INVALID_STR_ID);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

TEST(StringDictionary, BuildTranslationMap) {
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_string_dict =
      std::make_shared<StringDictionary>(dict_ref1, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_string_dict =
      std::make_shared<StringDictionary>(dict_ref2, g_cache_string_hash);

  // Prep: insert g_op_count strings into source_string_dict
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  source_string_dict->getOrAddBulk(strings, string_ids.data());
  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };

  {
    // First try to translate to empty dictionary.
    // Should get back all INVALID_STR_IDs

    const auto translated_ids = StringDictionaryTranslator::buildDictionaryTranslationMap(
        source_string_dict, dest_string_dict, dummy_callback);
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict->storageEntryCount());
    for (size_t idx = 0; idx < num_ids; ++idx) {
      ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
    }
  }

  {
    // Now insert the same strings inserted into source_string_dict,
    // but in reversed order and missing the first 2 strings in the source
    // dictionary. Make sure they translate correctly, and that the missing
    // strings translate to INVALID_STR_ID
    std::vector<std::string> reversed_strings;
    constexpr int32_t num_missing_strings{2};
    ASSERT_LE(num_missing_strings, g_op_count);
    std::vector<int32_t> reversed_string_ids(g_op_count - num_missing_strings);
    for (int i = 0; i < g_op_count - num_missing_strings; ++i) {
      reversed_strings.emplace_back(std::to_string(g_op_count - i - 1));
    }
    dest_string_dict->getOrAddBulk(reversed_strings, reversed_string_ids.data());
    ASSERT_EQ(dest_string_dict->storageEntryCount(), reversed_strings.size());
    const auto translated_ids = StringDictionaryTranslator::buildDictionaryTranslationMap(
        source_string_dict, dest_string_dict, dummy_callback);
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, static_cast<size_t>(g_op_count));
    ASSERT_EQ(num_ids,
              reversed_strings.size() + static_cast<size_t>(num_missing_strings));
    for (int32_t idx = 0; idx < static_cast<int32_t>(num_ids); ++idx) {
      if (idx < num_missing_strings) {
        ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
      } else {
        ASSERT_EQ(translated_ids[idx], g_op_count - idx - 1);
      }
    }
  }
}

TEST(NestedStringDictionary, AddAndGet) {
  std::vector<std::shared_ptr<StringDictionary>> dicts;
  for (int dict_no = 0; dict_no < 10; ++dict_no) {
    if (dict_no) {
      dicts.emplace_back(
          std::make_shared<StringDictionary>(dicts.back(), -1, g_cache_string_hash));
    } else {
      dicts.emplace_back(std::make_shared<StringDictionary>(DictRef{-1, dict_no + 1},
                                                            g_cache_string_hash));
    }

    auto& dict = *dicts.back();
    auto entries = dict.entryCount();
    ASSERT_EQ(entries, (dicts.size() - 1) * 2);
    for (int str_id = 0; str_id < static_cast<int>(dicts.size() * 2); ++str_id) {
      auto str_val = std::to_string(str_id);
      if (str_id < static_cast<int>(entries)) {
        ASSERT_EQ(dict.getIdOfString(str_val), str_id);
        ASSERT_EQ(dict.getString(str_id), str_val);
      } else {
        ASSERT_EQ(dict.getIdOfString(str_val), StringDictionary::INVALID_STR_ID);
      }
      ASSERT_EQ(dict.getOrAdd(str_val), str_id);
      ASSERT_EQ(dict.getOrAdd(str_val), str_id);
      ASSERT_EQ(dict.getIdOfString(str_val), str_id);
      ASSERT_EQ(dict.getString(str_id), str_val);
    }
  }
}

TEST(NestedStringDictionary, AddAndGetModifiedBase) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);
  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict2->getOrAdd("str1"), 0);
  ASSERT_EQ(dict2->getOrAdd("str2"), 1);
  ASSERT_EQ(dict2->getOrAdd("str3"), 2);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getIdOfString("str4"s), StringDictionary::INVALID_STR_ID);
  ASSERT_EQ(dict2->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getIdOfString("str4"s), 3);
  auto dict3 = std::make_shared<StringDictionary>(dict2, 2, g_cache_string_hash);
  ASSERT_EQ(dict3->getIdOfString("str1"s), 0);
  ASSERT_EQ(dict3->getIdOfString("str2"s), 1);
  ASSERT_EQ(dict3->getIdOfString("str3"s), StringDictionary::INVALID_STR_ID);
  ASSERT_EQ(dict3->getIdOfString("str4"s), StringDictionary::INVALID_STR_ID);
  ASSERT_EQ(dict3->getOrAdd("str3"), 2);
  ASSERT_EQ(dict3->getOrAdd("str4"), 3);
  ASSERT_EQ(dict3->getIdOfString("str3"s), 2);
  ASSERT_EQ(dict3->getIdOfString("str4"s), 3);
}

TEST(NestedStringDictionary, GetOrAddTransient) {
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict =
      std::make_shared<StringDictionary>(dict_ref, g_cache_string_hash);

  // Prep underlying dictionary data
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    if (i % 2 == 0) {
      strings.emplace_back(std::to_string(i));
    } else {
      strings.emplace_back(std::to_string(i * -1));
    }
  }
  string_dict->getOrAddBulk(strings, string_ids.data());

  // Now make proxy from dictionary

  StringDictionary string_dict_proxy(string_dict, string_dict->entryCount());

  {
    // First iteration is identical to first of the StringDictionary GetOrAddBulk test,
    // and results should be the same
    std::vector<std::string> strings;
    std::vector<int32_t> string_ids(g_op_count);
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    for (size_t i = 0; i < g_op_count; ++i) {
      string_ids[i] = string_dict_proxy.getOrAdd(strings[i]);
    }

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now make the odd idx strings new, should get back transient string ids for those
    std::vector<std::string> strings;
    std::vector<int32_t> string_ids(g_op_count);
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 4 + i));
      }
    }

    for (size_t i = 0; i < g_op_count; ++i) {
      string_ids[i] = string_dict_proxy.getOrAdd(strings[i]);
    }

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], string_dict_proxy.getBaseGeneration() + (i / 2));
      }
    }
  }
}

TEST(NestedStringDictionary, GetBulk) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);
  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  {
    std::vector<int> ids(5, -10);
    auto missing = dict1->getBulk(
        std::vector<std::string>{"str1"s, "str2"s, "str3"s, "str4"s, "str5"s},
        ids.data(),
        -1);
    ASSERT_EQ(missing, (size_t)1);
    ASSERT_EQ(ids, std::vector<int>({0, 1, 2, 3, StringDictionary::INVALID_STR_ID}));
  }

  {
    std::vector<int> ids(5, -10);
    auto missing = dict1->getBulk(
        std::vector<std::string>{"str1"s, "str2"s, "str3"s, "str4"s, "str5"s},
        ids.data(),
        2);
    ASSERT_EQ(missing, (size_t)3);
    ASSERT_EQ(ids,
              std::vector<int>({0,
                                1,
                                StringDictionary::INVALID_STR_ID,
                                StringDictionary::INVALID_STR_ID,
                                StringDictionary::INVALID_STR_ID}));
  }

  {
    std::vector<int> ids(5, -10);
    auto missing = dict2->getBulk(
        std::vector<std::string>{"str1"s, "str2"s, "str3"s, "str4"s, "str6"s},
        ids.data(),
        -1);
    ASSERT_EQ(missing, (size_t)1);
    ASSERT_EQ(ids, std::vector<int>({0, 1, 2, StringDictionary::INVALID_STR_ID, 4}));
  }

  {
    std::vector<int> ids(5, -10);
    auto missing = dict2->getBulk(
        std::vector<std::string>{"str1"s, "str2"s, "str3"s, "str5"s, "str6"s},
        ids.data(),
        4);
    ASSERT_EQ(missing, (size_t)1);
    ASSERT_EQ(ids, std::vector<int>({0, 1, 2, 3, StringDictionary::INVALID_STR_ID}));
  }

  {
    std::vector<int> ids(5, -10);
    auto missing = dict2->getBulk(
        std::vector<std::string>{"str1"s, "str2"s, "str3"s, "str4"s, "str5"s},
        ids.data(),
        2);
    ASSERT_EQ(missing, (size_t)3);
    ASSERT_EQ(ids,
              std::vector<int>({0,
                                1,
                                StringDictionary::INVALID_STR_ID,
                                StringDictionary::INVALID_STR_ID,
                                StringDictionary::INVALID_STR_ID}));
  }

  {
    std::vector<int> ids(2, -10);
    auto missing =
        dict2->getBulk(std::vector<std::string>{"str1"s, "str2"s}, ids.data(), -1);
    ASSERT_EQ(missing, (size_t)0);
    ASSERT_EQ(ids, std::vector<int>({0, 1}));
  }
}

TEST(NestedStringDictionary, GetOrAddBulk) {
  std::shared_ptr<StringDictionary> dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str3"), 1);
  ASSERT_EQ(dict1->getOrAdd("str5"), 2);
  ASSERT_EQ(dict1->getOrAdd("str7"), 3);
  ASSERT_EQ(dict1->getOrAdd("str9"), 4);

  bool old_enable_stringdict_parallel = g_enable_stringdict_parallel;
  bool old_cache_string_hash = g_cache_string_hash;
  ScopeGuard g([old_enable_stringdict_parallel, old_cache_string_hash]() {
    g_enable_stringdict_parallel = old_enable_stringdict_parallel;
    g_cache_string_hash = old_cache_string_hash;
  });
  for (bool enable_stringdict_parallel : {true, false}) {
    for (bool cache_string_hash : {true, false}) {
      g_enable_stringdict_parallel = enable_stringdict_parallel;
      g_cache_string_hash = cache_string_hash;
      StringDictionary dict2(dict1, 3, g_cache_string_hash);
      std::vector<int> ids(10, -10);
      // Only str1, str2, and str3 should be used from the base dict.
      dict2.getOrAddBulk(std::vector<std::string>({"str1"s,
                                                   "str2"s,
                                                   "str3"s,
                                                   "str4"s,
                                                   "str5"s,
                                                   "str6"s,
                                                   "str7"s,
                                                   "str8"s,
                                                   "str9"s,
                                                   "str10"s}),
                         ids.data());
      ASSERT_EQ(ids, std::vector<int>({0, 3, 1, 4, 2, 5, 6, 7, 8, 9}));
    }
  }
}

static std::shared_ptr<StringDictionary> create_and_fill_dictionary() {
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict =
      std::make_shared<StringDictionary>(dict_ref, g_cache_string_hash);

  // fill the dictionary
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    if (i % 2 == 0) {
      strings.emplace_back(std::to_string(i));
    } else {
      strings.emplace_back(std::to_string(i * -1));
    }
  }
  string_dict->getOrAddBulk(strings, string_ids.data());

  return string_dict;
}

TEST(NestedStringDictionary, GetOrAddTransientBulk) {
  auto string_dict = create_and_fill_dictionary();

  StringDictionary string_dict_proxy(string_dict, string_dict->entryCount());
  {
    // First iteration is identical to first of the StringDictionary GetOrAddBulk test,
    // and results should be the same
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    const auto string_ids = string_dict_proxy.getOrAddBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now make the odd idx strings new, should get back transient string ids for those
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 6 + i));
      }
    }

    const auto string_ids = string_dict_proxy.getOrAddBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], string_dict_proxy.getBaseGeneration() + (i / 2));
      }
    }
  }
}

TEST(NestedStringDictionary, GetTransientBulk) {
  auto string_dict = create_and_fill_dictionary();

  StringDictionary string_dict_proxy(string_dict, string_dict->entryCount());
  {
    // First iteration is identical to first of the StryingDictionaryProxy
    // GetOrAddTransientBulk test, and results should be the same
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    const auto string_ids = string_dict_proxy.getBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }
  {
    // Second iteration - even idxs exist, odd do not and should return INVALID_STR_ID
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 8 + i));
      }
    }

    const auto string_ids = string_dict_proxy.getBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

class CheckStringsCallback : public StringDictionary::StringCallback {
 public:
  CheckStringsCallback(std::set<std::string> expected) : expected_(expected) {}

  void operator()(std::string const& str, int32_t const string_id) override {
    ASSERT_TRUE(expected_.count(str));
    expected_.erase(str);
  }

  void operator()(std::string_view const sv, int32_t const string_id) override {
    this->operator()(std::string(sv), string_id);
  }

  bool empty() const { return expected_.empty(); }

 private:
  std::set<std::string> expected_;
};

TEST(NestedStringDictionary, EachStringSerially) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);
  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  {
    CheckStringsCallback callback({"str1"s, "str2"s, "str3"s, "str4"});
    dict1->eachStringSerially(callback);
    ASSERT_TRUE(callback.empty());
  }

  {
    CheckStringsCallback callback({"str1"s, "str2"s, "str3"s, "str5", "str6"s});
    dict2->eachStringSerially(callback);
    ASSERT_TRUE(callback.empty());
  }

  {
    CheckStringsCallback callback({"str1"s, "str2"s, "str3"s, "str5"});
    dict2->eachStringSerially(4, callback);
    ASSERT_TRUE(callback.empty());
  }

  {
    CheckStringsCallback callback({"str1"s, "str2"s});
    dict2->eachStringSerially(2, callback);
    ASSERT_TRUE(callback.empty());
  }
}

TEST(NestedStringDictionary, CopyStrings) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);

  ASSERT_EQ(dict1->copyStrings(), std::vector<std::string>({"str1"s, "str2"s, "str3"s}));
  ASSERT_EQ(dict1->copyStrings(10),
            std::vector<std::string>({"str1"s, "str2"s, "str3"s}));
  ASSERT_EQ(dict1->copyStrings(2), std::vector<std::string>({"str1"s, "str2"s}));

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  ASSERT_EQ(dict1->copyStrings(),
            std::vector<std::string>({"str1"s, "str2"s, "str3"s, "str4"s}));
  ASSERT_EQ(dict1->copyStrings(3), std::vector<std::string>({"str1"s, "str2"s, "str3"s}));

  ASSERT_EQ(dict2->copyStrings(),
            std::vector<std::string>({"str1"s, "str2"s, "str3"s, "str5"s, "str6"s}));
  ASSERT_EQ(dict2->copyStrings(10),
            std::vector<std::string>({"str1"s, "str2"s, "str3"s, "str5"s, "str6"s}));
  ASSERT_EQ(dict2->copyStrings(4),
            std::vector<std::string>({"str1"s, "str2"s, "str3"s, "str5"s}));
  ASSERT_EQ(dict2->copyStrings(2), std::vector<std::string>({"str1"s, "str2"s}));

  ASSERT_EQ(dict2->getOrAdd("str7"), 5);

  ASSERT_EQ(
      dict2->copyStrings(),
      std::vector<std::string>({"str1"s, "str2"s, "str3"s, "str5"s, "str6"s, "str7"s}));
}

TEST(NestedStringDictionary, CopyStringsParallel) {
  constexpr int STR_COUNT = 20'000;

  std::vector<std::string> strings1;
  std::vector<std::string> strings2;
  for (int i = 0; i < STR_COUNT; ++i) {
    strings1.push_back(std::to_string(i));
    strings2.push_back(std::to_string(STR_COUNT + i));
  }
  strings2.insert(strings2.begin(), strings1.begin(), strings1.end());

  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  dict1->getOrAddBulk(strings1);

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  dict2->getOrAddBulk(strings2);

  ASSERT_EQ(dict1->copyStrings(), strings1);
  ASSERT_EQ(dict2->copyStrings(), strings2);

  dict1->getOrAddBulk(strings2);

  ASSERT_EQ(dict1->copyStrings(), strings2);
}

TEST(NestedStringDictionary, IsLike) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);

  ASSERT_EQ(dict1->getLike("str%", true, false, '\\'), std::vector<int>({0, 1, 2}));
  ASSERT_EQ(dict1->getLike("str%", true, false, '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict1->getLike("str[124]", true, false, '\\'), std::vector<int>({0, 1}));

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  ASSERT_EQ(dict1->getLike("str%", true, false, '\\'), std::vector<int>({0, 1, 2, 3}));
  ASSERT_EQ(dict1->getLike("str%", true, false, '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict1->getLike("str[124]", true, false, '\\'), std::vector<int>({0, 1, 3}));

  ASSERT_EQ(dict2->getLike("str%", true, false, '\\'), std::vector<int>({0, 1, 2, 3, 4}));
  ASSERT_EQ(dict2->getLike("str%", true, false, '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict2->getLike("str[12467]", true, false, '\\'), std::vector<int>({0, 1, 4}));

  ASSERT_EQ(dict1->getOrAdd("str6"), 4);
  ASSERT_EQ(dict2->getOrAdd("str7"), 5);

  ASSERT_EQ(dict2->getLike("str[12467]", true, false, '\\'),
            std::vector<int>({0, 1, 4, 5}));
}

void sortAndCompare(std::vector<int> actual, const std::vector<int>& expected) {
  std::sort(actual.begin(), actual.end(), std::less<int>());
  ASSERT_EQ(actual, expected);
}

TEST(NestedStringDictionary, GetEquals) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);

  sortAndCompare(dict1->getCompare("str1", "="), {0});
  sortAndCompare(dict1->getCompare("str2", "="), {1});
  sortAndCompare(dict1->getCompare("str5", "="), {});
  sortAndCompare(dict1->getCompare("str2", "<>"), {0, 2});
  sortAndCompare(dict1->getCompare("str2", "=", 1), {});
  sortAndCompare(dict1->getCompare("str2", "<>", 1), {0});

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  sortAndCompare(dict1->getCompare("str4", "="), {3});
  sortAndCompare(dict1->getCompare("str4", "=", 3), {});
  sortAndCompare(dict1->getCompare("str2", "<>"), {0, 2, 3});
  sortAndCompare(dict1->getCompare("str2", "<>", 3), {0, 2});

  sortAndCompare(dict2->getCompare("str4", "="), {});
  sortAndCompare(dict2->getCompare("str5", "="), {3});
  sortAndCompare(dict2->getCompare("str5", "=", 3), {});
  sortAndCompare(dict2->getCompare("str5", "<>"), {0, 1, 2, 4});
  sortAndCompare(dict2->getCompare("str5", "<>", 3), {0, 1, 2});

  ASSERT_EQ(dict2->getOrAdd("str7"), 5);

  sortAndCompare(dict2->getCompare("str5", "<>"), {0, 1, 2, 4, 5});
  sortAndCompare(dict2->getCompare("str7", "="), {5});
}

TEST(NestedStringDictionary, GetCompare) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);

  sortAndCompare(dict1->getCompare("str2", "<"), {0});
  sortAndCompare(dict1->getCompare("str2", "<="), {0, 1});
  sortAndCompare(dict1->getCompare("str2", "<=", 1), {0});
  sortAndCompare(dict1->getCompare("str2", ">"), {2});
  sortAndCompare(dict1->getCompare("str2", ">="), {1, 2});
  sortAndCompare(dict1->getCompare("str2", ">=", 2), {1});
  sortAndCompare(dict1->getCompare("str2", "="), {1});
  sortAndCompare(dict1->getCompare("str2", "<>"), {0, 2});

  sortAndCompare(dict1->getCompare("str11", "<"), {0});
  sortAndCompare(dict1->getCompare("str11", "<="), {0});
  sortAndCompare(dict1->getCompare("str11", ">"), {1, 2});
  sortAndCompare(dict1->getCompare("str11", ">="), {1, 2});
  sortAndCompare(dict1->getCompare("str11", "<>"), {0, 1, 2});
  sortAndCompare(dict1->getCompare("str11", "="), {});

  sortAndCompare(dict1->getCompare("str0", "<"), {});
  sortAndCompare(dict1->getCompare("str0", "<="), {});
  sortAndCompare(dict1->getCompare("str0", ">"), {0, 1, 2});
  sortAndCompare(dict1->getCompare("str0", ">="), {0, 1, 2});
  sortAndCompare(dict1->getCompare("str0", "="), {});
  sortAndCompare(dict1->getCompare("str0", "<>"), {0, 1, 2});

  sortAndCompare(dict1->getCompare("str4", "<"), {0, 1, 2});
  sortAndCompare(dict1->getCompare("str4", "<="), {0, 1, 2});
  sortAndCompare(dict1->getCompare("str4", ">"), {});
  sortAndCompare(dict1->getCompare("str4", ">="), {});
  sortAndCompare(dict1->getCompare("str4", "="), {});
  sortAndCompare(dict1->getCompare("str4", "<>"), {0, 1, 2});

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);
  ASSERT_EQ(dict2->getOrAdd("str7"), 5);

  sortAndCompare(dict1->getCompare("str2", "<"), {0});
  sortAndCompare(dict1->getCompare("str2", "<="), {0, 1});
  sortAndCompare(dict1->getCompare("str2", ">"), {2, 3});
  sortAndCompare(dict1->getCompare("str2", ">="), {1, 2, 3});
  sortAndCompare(dict1->getCompare("str2", "="), {1});
  sortAndCompare(dict1->getCompare("str2", "<>"), {0, 2, 3});

  sortAndCompare(dict2->getCompare("str6", "<"), {0, 1, 2, 3});
  sortAndCompare(dict2->getCompare("str6", "<", 3), {0, 1, 2});
  sortAndCompare(dict2->getCompare("str6", "<", 1), {0});
  sortAndCompare(dict2->getCompare("str6", "<="), {0, 1, 2, 3, 4});
  sortAndCompare(dict2->getCompare("str6", "<=", 3), {0, 1, 2});
  sortAndCompare(dict2->getCompare("str6", "<=", 1), {0});
  sortAndCompare(dict2->getCompare("str6", ">"), {5});
  sortAndCompare(dict2->getCompare("str6", ">", 3), {});
  sortAndCompare(dict2->getCompare("str6", ">", 1), {});
  sortAndCompare(dict2->getCompare("str6", ">="), {4, 5});
  sortAndCompare(dict2->getCompare("str6", ">=", 3), {});
  sortAndCompare(dict2->getCompare("str6", ">=", 1), {});
  sortAndCompare(dict2->getCompare("str6", ">=", 5), {4});
  sortAndCompare(dict2->getCompare("str6", "="), {4});
  sortAndCompare(dict2->getCompare("str6", "=", 3), {});
  sortAndCompare(dict2->getCompare("str6", "=", 1), {});
  sortAndCompare(dict2->getCompare("str6", "<>"), {0, 1, 2, 3, 5});
  sortAndCompare(dict2->getCompare("str6", "<>", 3), {0, 1, 2});
  sortAndCompare(dict2->getCompare("str6", "<>", 1), {0});
}

TEST(NestedStringDictionary, GetRegexpLike) {
  auto dict1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str1"), 0);
  ASSERT_EQ(dict1->getOrAdd("str2"), 1);
  ASSERT_EQ(dict1->getOrAdd("str3"), 2);

  ASSERT_EQ(dict1->getRegexpLike("str.", '\\'), std::vector<int>({0, 1, 2}));
  ASSERT_EQ(dict1->getRegexpLike("str.", '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict1->getRegexpLike("str[124]", '\\'), std::vector<int>({0, 1}));

  auto dict2 = std::make_shared<StringDictionary>(dict1, -1, g_cache_string_hash);
  ASSERT_EQ(dict1->getOrAdd("str4"), 3);
  ASSERT_EQ(dict2->getOrAdd("str5"), 3);
  ASSERT_EQ(dict2->getOrAdd("str6"), 4);

  ASSERT_EQ(dict1->getRegexpLike("str.", '\\'), std::vector<int>({0, 1, 2, 3}));
  ASSERT_EQ(dict1->getRegexpLike("str.", '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict1->getRegexpLike("str[124]", '\\'), std::vector<int>({0, 1, 3}));

  ASSERT_EQ(dict2->getRegexpLike("str.", '\\'), std::vector<int>({0, 1, 2, 3, 4}));
  ASSERT_EQ(dict2->getRegexpLike("str.", '\\', 2), std::vector<int>({0, 1}));
  ASSERT_EQ(dict2->getRegexpLike("str[12467]", '\\'), std::vector<int>({0, 1, 4}));

  ASSERT_EQ(dict1->getOrAdd("str6"), 4);
  ASSERT_EQ(dict2->getOrAdd("str7"), 5);

  ASSERT_EQ(dict2->getRegexpLike("str[12467]", '\\'), std::vector<int>({0, 1, 4, 5}));
}

TEST(NestedStringDictionary, BuildTranslationMap_EmptyDict) {
  auto source1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  auto source2 = std::make_shared<StringDictionary>(source1, -1, g_cache_string_hash);

  auto dest1 =
      std::make_shared<StringDictionary>(DictRef{-1, 2}, -1, g_cache_string_hash);
  dest1->getOrAdd("str1");
  auto dest2 = std::make_shared<StringDictionary>(dest1, -1, g_cache_string_hash);
  dest2->getOrAdd("str2");

  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };

  ASSERT_EQ(StringDictionaryTranslator::buildDictionaryTranslationMap(
                source2, dest2, dummy_callback),
            std::vector<int>({}));

  ASSERT_EQ(StringDictionaryTranslator::buildDictionaryTranslationMap(
                dest2, source2, dummy_callback),
            std::vector<int>(
                {StringDictionary::INVALID_STR_ID, StringDictionary::INVALID_STR_ID}));
}

TEST(NestedStringDictionary, BuildTranslationMap) {
  auto source1 =
      std::make_shared<StringDictionary>(DictRef{-1, 1}, -1, g_cache_string_hash);
  ASSERT_EQ(source1->getOrAdd("source1"), 0);
  ASSERT_EQ(source1->getOrAdd("common1"), 1);
  ASSERT_EQ(source1->getOrAdd("source2"), 2);
  ASSERT_EQ(source1->getOrAdd("common2"), 3);
  auto source2 = std::make_shared<StringDictionary>(source1, -1, g_cache_string_hash);
  ASSERT_EQ(source2->getOrAdd("source3"), 4);
  ASSERT_EQ(source2->getOrAdd("common3"), 5);
  ASSERT_EQ(source2->getOrAdd("source4"), 6);
  ASSERT_EQ(source2->getOrAdd("common4"), 7);
  // Add some string that should be ignored on translation.
  ASSERT_EQ(source1->getOrAdd("source4"), 4);
  ASSERT_EQ(source1->getOrAdd("common4"), 5);
  ASSERT_EQ(source1->getOrAdd("source3"), 6);
  ASSERT_EQ(source1->getOrAdd("common3"), 7);

  auto dest1 =
      std::make_shared<StringDictionary>(DictRef{-1, 2}, -1, g_cache_string_hash);
  ASSERT_EQ(dest1->getOrAdd("common1"), 0);
  ASSERT_EQ(dest1->getOrAdd("common2"), 1);
  ASSERT_EQ(dest1->getOrAdd("dest1"), 2);
  ASSERT_EQ(dest1->getOrAdd("dest2"), 3);
  auto dest2 = std::make_shared<StringDictionary>(dest1, -1, g_cache_string_hash);
  ASSERT_EQ(dest2->getOrAdd("dest3"), 4);
  ASSERT_EQ(dest2->getOrAdd("dest4"), 5);
  ASSERT_EQ(dest2->getOrAdd("common3"), 6);
  ASSERT_EQ(dest2->getOrAdd("common4"), 7);
  // Add some string that should be ignored on translation.
  ASSERT_EQ(dest1->getOrAdd("common3"), 4);
  ASSERT_EQ(dest1->getOrAdd("common4"), 5);
  ASSERT_EQ(dest1->getOrAdd("dest3"), 6);
  ASSERT_EQ(dest1->getOrAdd("dest4"), 7);

  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };

  CHECK_EQ(dest2->getIdOfString(std::string("common3")), 6);
  CHECK_EQ(dest2->getIdOfString(std::string("common4")), 7);

  ASSERT_EQ(StringDictionaryTranslator::buildDictionaryTranslationMap(
                source2, dest2, dummy_callback),
            std::vector<int>({StringDictionary::INVALID_STR_ID,
                              0,
                              StringDictionary::INVALID_STR_ID,
                              1,
                              StringDictionary::INVALID_STR_ID,
                              6,
                              StringDictionary::INVALID_STR_ID,
                              7}));

  ASSERT_EQ(StringDictionaryTranslator::buildDictionaryTranslationMap(
                dest2, source2, dummy_callback),
            std::vector<int>({1,
                              3,
                              StringDictionary::INVALID_STR_ID,
                              StringDictionary::INVALID_STR_ID,
                              StringDictionary::INVALID_STR_ID,
                              StringDictionary::INVALID_STR_ID,
                              5,
                              7}));
}

TEST(StringDictionaryProxy, BuildIntersectionTranslationMapToOtherProxy) {
  // Use existing dictionary from GetBulk
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_string_dict =
      std::make_shared<StringDictionary>(dict_ref1, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_string_dict =
      std::make_shared<StringDictionary>(dict_ref2, g_cache_string_hash);

  // Prep: insert g_op_count strings into source_string_dict
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  source_string_dict->getOrAddBulk(strings, string_ids.data());

  {
    // First try to translate to empty dictionary.
    // Should get back all INVALID_STR_IDs
    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                dest_string_dict->storageEntryCount());
    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildIntersectionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get());
    ASSERT_FALSE(str_proxy_translation_map.empty());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
    for (size_t idx = 0; idx < num_ids; ++idx) {
      ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
    }
  }

  {
    // Add transient entries to source proxy
    constexpr int32_t num_missing_ids{10};
    ASSERT_LE(num_missing_ids, g_op_count);
    const int32_t num_dest_strings{g_op_count - num_missing_ids};
    std::vector<int32_t> dest_string_ids(num_dest_strings);
    std::vector<std::string> dest_strings;
    dest_strings.reserve(num_dest_strings);
    for (int i = 0; i < num_dest_strings; ++i) {
      dest_strings.emplace_back(std::to_string(i));
    }
    dest_string_dict->getOrAddBulk(dest_strings, dest_string_ids.data());

    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                dest_string_dict->storageEntryCount());

    constexpr int32_t num_source_proxy_transient_ids{64};
    constexpr int32_t num_dest_proxy_transient_ids{32};
    constexpr int32_t proxy_transient_start{g_op_count * 2};
    for (int i = proxy_transient_start;
         i < proxy_transient_start + num_source_proxy_transient_ids;
         ++i) {
      source_string_dict_proxy->getOrAdd(std::to_string(i));
    }
    for (int i = proxy_transient_start;
         i < proxy_transient_start + num_dest_proxy_transient_ids;
         ++i) {
      dest_string_dict_proxy->getOrAdd(std::to_string(i));
    }

    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildIntersectionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get());
    ASSERT_FALSE(str_proxy_translation_map.empty());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
    for (int32_t idx = 1; idx < static_cast<int32_t>(num_ids); ++idx) {
      const std::string source_string = source_string_dict_proxy->getString(idx);
      const int32_t source_string_as_int = std::stoi(source_string);
      const bool should_be_missing =
          !((source_string_as_int >= 0 && source_string_as_int < num_dest_strings) ||
            (source_string_as_int >= (g_op_count * 2) &&
             source_string_as_int < (g_op_count * 2 + num_dest_proxy_transient_ids)));
      if (translated_ids[idx] != StringDictionary::INVALID_STR_ID) {
        ASSERT_FALSE(should_be_missing);
        const std::string dest_string =
            dest_string_dict_proxy->getString(translated_ids[idx]);
        ASSERT_EQ(source_string, dest_string);

      } else {
        ASSERT_TRUE(should_be_missing);
      }
    }
  }
}

TEST(StringDictionaryProxy, BuildUnionTranslationMapToEmptyProxy) {
  // Todo(todd): Migrate this and intersection translation tests to use
  // approach and methods in BuildUnionTranslationMapToPartialOverlapProxy
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_string_dict =
      std::make_shared<StringDictionary>(dict_ref1, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_string_dict =
      std::make_shared<StringDictionary>(dict_ref2, g_cache_string_hash);

  // Prep: insert g_op_count strings into source_string_dict
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  source_string_dict->getOrAddBulk(strings, string_ids.data());

  {
    // First try to union translate to empty dictionary.
    // All strings should end up as transient entries in
    // destination proxy
    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                dest_string_dict->storageEntryCount());
    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildUnionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get());
    ASSERT_FALSE(str_proxy_translation_map.empty());
    ASSERT_EQ(str_proxy_translation_map.numUntranslatedStrings(), strings.size());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
    for (size_t idx = 0; idx < num_ids; ++idx) {
      const auto string_id = static_cast<int32_t>(idx);
      if (string_id == -1) {
        ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
      } else {
        ASSERT_EQ(dest_string_dict_proxy->getString(translated_ids[idx]),
                  strings[string_id]);
      }
    }
  }
}

std::vector<std::string> add_strings_numeric_range(std::shared_ptr<StringDictionary> sd,
                                                   const size_t num_vals,
                                                   const int32_t start_val) {
  CHECK_GE(start_val, 0);
  const auto end_val = start_val + static_cast<int32_t>(num_vals);
  std::vector<std::string> strings(num_vals);
  for (int32_t int_val = start_val; int_val < end_val; ++int_val) {
    strings[int_val - start_val] = std::to_string(int_val);
  }
  std::vector<int32_t> string_ids(num_vals);
  sd->getOrAddBulk(strings, string_ids.data());
  return strings;
}

std::vector<std::string> add_strings_numeric_range(StringDictionaryProxy& sdp,
                                                   const size_t num_vals,
                                                   const int32_t start_val) {
  CHECK_GE(start_val, sdp.getBaseGeneration());
  std::vector<std::string> strings(num_vals);
  const auto end_val = start_val + static_cast<int32_t>(num_vals);
  for (int32_t int_val = start_val; int_val < end_val; ++int_val) {
    strings[int_val - start_val] = std::to_string(-int_val);
  }
  sdp.getOrAddBulk(strings);
  return strings;
}

size_t calc_expected_untranslated_strings(const size_t num_source_entries,
                                          const size_t num_dest_entries,
                                          const int32_t source_start_val,
                                          const int32_t dest_start_val) {
  const int32_t source_abs_min = std::abs(source_start_val);  // 0
  const int32_t source_abs_max =
      std::abs(source_start_val) + static_cast<int32_t>(num_source_entries);  // 10
  const int32_t dest_abs_min = std::abs(dest_start_val);                      // 7
  const int32_t dest_abs_max =
      std::abs(dest_start_val) + static_cast<int32_t>(num_dest_entries);  // 12
  return static_cast<size_t>(std::max(dest_abs_min - source_abs_min, 0) +
                             std::max(source_abs_max - dest_abs_max, 0));
}

void verify_translation(const StringDictionaryProxy& source_proxy,
                        const StringDictionaryProxy& dest_proxy,
                        const StringDictionaryProxy::IdMap& id_map,
                        const std::vector<std::string>& persisted_source_strings,
                        const std::vector<std::string>& transient_source_strings,
                        const std::vector<std::string>& persisted_dest_strings,
                        const std::vector<std::string>& transient_dest_strings,
                        const bool is_union) {
  std::unordered_set<std::string> dest_string_set;
  for (const auto& str : persisted_dest_strings) {
    ASSERT_TRUE(dest_string_set.insert(str).second);
  }
  for (const auto& str : transient_dest_strings) {
    ASSERT_TRUE(dest_string_set.insert(str).second);
  }
  for (const auto& persisted_source_str : persisted_source_strings) {
    const int32_t source_str_id = source_proxy.getIdOfString(persisted_source_str);
    ASSERT_GE(source_str_id, 0);
    const bool str_was_in_dest_proxy =
        dest_string_set.find(persisted_source_str) != dest_string_set.end();
    const auto translated_id = id_map[source_str_id];
    if (str_was_in_dest_proxy) {
      ASSERT_GE(translated_id, 0);
      const auto dest_str = dest_proxy.getString(translated_id);
      ASSERT_EQ(persisted_source_str, dest_str);
    } else {
      if (is_union) {
        ASSERT_GE(translated_id, dest_proxy.getBaseGeneration());
        const auto dest_str = dest_proxy.getString(translated_id);
        ASSERT_EQ(persisted_source_str, dest_str);
      } else {
        ASSERT_EQ(translated_id, StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

TEST(StringDictionaryProxy, BuildUnionTranslationMapToPartialOverlapProxy) {
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_sd =
      std::make_shared<StringDictionary>(dict_ref1, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_sd =
      std::make_shared<StringDictionary>(dict_ref2, g_cache_string_hash);

  constexpr size_t num_source_persisted_entries{10};
  constexpr size_t num_dest_persisted_entries{5};
  constexpr int32_t source_persisted_start_val{0};
  constexpr int32_t dest_persisted_start_val{7};

  constexpr size_t num_source_transient_entries{10};
  constexpr size_t num_dest_transient_entries{5};
  constexpr int32_t source_transient_start_val{12};
  constexpr int32_t dest_transient_start_val{19};
  const auto persisted_source_strings = add_strings_numeric_range(
      source_sd, num_source_persisted_entries, source_persisted_start_val);
  ASSERT_EQ(source_sd->storageEntryCount(), num_source_persisted_entries);

  const auto persisted_dest_strings = add_strings_numeric_range(
      dest_sd, num_dest_persisted_entries, dest_persisted_start_val);
  ASSERT_EQ(dest_sd->storageEntryCount(), num_dest_persisted_entries);

  StringDictionaryProxy source_sdp(source_sd, source_sd->storageEntryCount());
  StringDictionaryProxy dest_sdp(dest_sd, dest_sd->storageEntryCount());
  const auto transient_source_strings = add_strings_numeric_range(
      source_sdp, num_source_transient_entries, source_transient_start_val);
  ASSERT_EQ(source_sdp.getBaseDictionary()->getDictId(), 1);
  ASSERT_EQ(source_sdp.getBaseDictionary()->storageEntryCount(),
            num_source_persisted_entries);
  ASSERT_EQ(source_sdp.transientEntryCount(), num_source_transient_entries);

  const auto transient_dest_strings = add_strings_numeric_range(
      dest_sdp, num_dest_transient_entries, dest_transient_start_val);
  ASSERT_EQ(dest_sdp.getBaseDictionary()->getDictId(), 2);
  ASSERT_EQ(dest_sdp.getBaseDictionary()->storageEntryCount(),
            num_dest_persisted_entries);
  ASSERT_EQ(dest_sdp.transientEntryCount(), num_dest_transient_entries);

  const auto id_map = source_sdp.buildUnionTranslationMapToOtherProxy(&dest_sdp);
  const size_t expected_num_untranslated_strings =
      calc_expected_untranslated_strings(num_source_persisted_entries,
                                         num_dest_persisted_entries,
                                         source_persisted_start_val,
                                         dest_persisted_start_val) +
      calc_expected_untranslated_strings(num_source_transient_entries,
                                         num_dest_transient_entries,
                                         source_transient_start_val,
                                         dest_transient_start_val);
  ASSERT_EQ(id_map.size(), num_source_persisted_entries + num_source_transient_entries);
  ASSERT_EQ(id_map.numNonTransients(), num_source_persisted_entries);
  ASSERT_EQ(id_map.numTransients(), num_source_transient_entries);
  ASSERT_EQ(id_map.numUntranslatedStrings(), expected_num_untranslated_strings);

  verify_translation(source_sdp,
                     dest_sdp,
                     id_map,
                     persisted_source_strings,
                     transient_source_strings,
                     persisted_dest_strings,
                     transient_dest_strings,
                     true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()(
      "enable-string-dict-hash-cache",
      po::value<bool>(&g_cache_string_hash)
          ->default_value(g_cache_string_hash)
          ->implicit_value(true),
      "Cache string hash values in the string dictionary server during import.");
  desc.add_options()("cpu-only", "ignored option");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  logger::init(log_options);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
