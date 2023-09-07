/*
 * Copyright 2021 OmniSci, Inc.
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

#ifndef STRINGDICTIONARY_STRINGDICTIONARY_H
#define STRINGDICTIONARY_STRINGDICTIONARY_H

#include "../Shared/mapd_shared_mutex.h"
#include "DictRef.h"
#include "DictionaryCache.hpp"

#include <functional>
#include <future>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#define USE_LEGACY_STR_DICT

extern bool g_enable_stringdict_parallel;

namespace legacy {
class StringDictionary;
}

namespace fast {
class StringDictionary;
}

#ifdef USE_LEGACY_STR_DICT
using StringDictionary = legacy::StringDictionary;
#else
using StringDictionary = fast::StringDictionary;
#endif

using StringLookupCallback = std::function<bool(std::string_view, int32_t string_id)>;

class StringDictionaryTranslator {
 public:
  static std::vector<int32_t> buildDictionaryTranslationMap(
      const std::shared_ptr<StringDictionary> source_dict,
      const std::shared_ptr<StringDictionary> dest_dict,
      StringLookupCallback const& dest_transient_lookup_callback);

  static size_t buildDictionaryTranslationMap(
      const StringDictionary* source_dict,
      const StringDictionary* dest_dict,
      int32_t* translated_ids,
      const int64_t source_generation,
      const int64_t dest_generation,
      const bool dest_has_transients,
      StringLookupCallback const& dest_transient_lookup_callback);

 private:
  StringDictionaryTranslator() {}
};

class StringLocalCallback;

namespace legacy {

class StringDictionary {
 public:
  StringDictionary(const DictRef& dict_ref,
                   const bool materializeHashes = false,
                   size_t initial_capacity = 256);
  StringDictionary(std::shared_ptr<StringDictionary> base_dict,
                   const int64_t generation = -1,
                   const bool materializeHashes = false,
                   size_t initial_capacity = 0);
  ~StringDictionary() noexcept;

  int32_t getDbId() const noexcept;
  int32_t getDictId() const noexcept;

  StringDictionary* getBaseDictionary() const noexcept { return base_dict_.get(); }
  int64_t getBaseGeneration() const noexcept { return base_generation_; }

  class StringCallback {
   public:
    virtual ~StringCallback() = default;
    virtual void operator()(std::string const&, int32_t const string_id) = 0;
    virtual void operator()(std::string_view const, int32_t const string_id) = 0;
  };

  // Functors passed to eachStringSerially() must derive from StringCallback.
  // Each std::string const& (if isClient()) or std::string_view (if !isClient())
  // plus string_id is passed to the callback functor.
  void eachStringSerially(int64_t const generation, StringCallback&) const;
  void eachStringSerially(StringCallback&) const;
  friend class ::StringLocalCallback;

  int32_t getOrAdd(const std::string_view& str) noexcept;
  template <class T, class String>
  size_t getBulk(const std::vector<String>& string_vec, T* encoded_vec) const;
  template <class T, class String>
  size_t getBulk(const std::vector<String>& string_vec,
                 T* encoded_vec,
                 const int64_t generation) const;
  template <class T, class String>
  void getOrAddBulk(const std::vector<String>& string_vec, T* encoded_vec);
  template <class String>
  std::vector<int32_t> getOrAddBulk(const std::vector<String>& string_vec);
  template <class String>
  std::vector<int32_t> getBulk(const std::vector<String>& string_vec);
  template <class String>
  int32_t getIdOfString(const String&) const;
  std::string getString(int32_t string_id) const;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t storageEntryCount() const;
  size_t entryCount() const;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape,
                               const size_t generation) const;

  std::vector<int32_t> getCompare(const std::string& pattern,
                                  const std::string& comp_operator,
                                  const size_t generation);

  std::vector<int32_t> getRegexpLike(const std::string& pattern,
                                     const char escape,
                                     const size_t generation) const;

  std::vector<std::string> copyStrings(int64_t generation = -1) const;

  static constexpr int32_t INVALID_STR_ID = -1;
  static constexpr size_t MAX_STRLEN = (1 << 15) - 1;
  static constexpr size_t MAX_STRCOUNT = (1U << 31) - 1;

 private:
  struct StringIdxEntry {
    uint64_t off : 48;
    uint64_t size : 16;
  };

  // In the compare_cache_value_t index represents the index of the sorted cache.
  // The diff component represents whether the index the cache is pointing to is equal to
  // the pattern it is cached for. We want to use diff so we don't have compare string
  // again when we are retrieving it from the cache.
  struct compare_cache_value_t {
    int32_t index;
    int32_t diff;
  };

  struct PayloadString {
    char* c_str_ptr;
    size_t size;
    bool canary;
  };

  bool fillRateIsHigh(const size_t num_strings) const noexcept;
  void increaseHashTableCapacity() noexcept;
  template <class String>
  void increaseHashTableCapacityFromStorageAndMemory(
      const size_t str_count,
      const size_t storage_high_water_mark,
      const std::vector<String>& input_strings,
      const std::vector<size_t>& string_memory_ids,
      const std::vector<uint32_t>& input_strings_hashes) noexcept;
  template <class String>
  void hashStrings(const std::vector<String>& string_vec,
                   std::vector<uint32_t>& hashes) const noexcept;

  template <class String>
  int32_t getIdOfString(const String&, const uint32_t hash) const;
  int32_t getUnlocked(const std::string_view sv) const noexcept;
  int32_t getUnlocked(const std::string_view sv, const uint32_t hash) const noexcept;
  int32_t getOwnedUnlocked(const std::string_view sv) const noexcept;
  int32_t getOwnedUnlocked(const std::string_view sv, const uint32_t hash) const noexcept;
  std::string getStringUnlocked(int32_t string_id) const noexcept;
  std::string getOwnedStringChecked(const int string_id) const noexcept;
  std::pair<char*, size_t> getOwnedStringBytesChecked(const int string_id) const noexcept;
  template <class T, class String>
  void getOrAddBulkParallel(const std::vector<String>& string_vec, T* encoded_vec);
  void copyStrings(int64_t string_id_start,
                   int64_t string_id_end,
                   std::vector<std::string>& out_vec) const;
  template <class String>
  uint32_t computeBucket(
      const uint32_t hash,
      const String& input_string,
      const std::vector<int32_t>& string_id_uint32_table) const noexcept;
  template <class String>
  uint32_t computeBucketFromStorageAndMemory(
      const uint32_t input_string_hash,
      const String& input_string,
      const std::vector<int32_t>& string_id_uint32_table,
      const size_t storage_high_water_mark,
      const std::vector<String>& input_strings,
      const std::vector<size_t>& string_memory_ids) const noexcept;
  uint32_t computeUniqueBucketWithHash(
      const uint32_t hash,
      const std::vector<int32_t>& string_id_uint32_table) noexcept;
  void checkAndConditionallyIncreasePayloadCapacity(const size_t write_length);
  void checkAndConditionallyIncreaseOffsetCapacity(const size_t write_length);

  template <class String>
  void appendToStorage(const String str) noexcept;
  template <class String>
  void appendToStorageBulk(const std::vector<String>& input_strings,
                           const std::vector<size_t>& string_memory_ids,
                           const size_t sum_new_strings_lengths) noexcept;
  PayloadString getStringFromStorage(const int string_id) const noexcept;
  std::string_view getStringFromStorageFast(const int string_id) const noexcept;
  void addPayloadCapacity(const size_t min_capacity_requested = 0) noexcept;
  void addOffsetCapacity(const size_t min_capacity_requested = 0) noexcept;
  void* addMemoryCapacity(void* addr,
                          size_t& mem_size,
                          const size_t min_capacity_requested = 0) noexcept;
  void invalidateInvertedIndex() noexcept;
  std::vector<int32_t> getEquals(std::string pattern,
                                 std::string comp_operator,
                                 size_t generation);
  void buildSortedCache();
  void sortCache(std::vector<int32_t>& cache);
  void mergeSortedCache(std::vector<int32_t>& temp_sorted_cache);

  int indexToId(int string_idx) const { return string_idx + base_generation_; }
  int idToIndex(int string_id) const { return string_id - base_generation_; }

  uint32_t hashById(int string_id) const { return hashByIndex(idToIndex(string_id)); }
  uint32_t hashByIndex(int string_idx) const { return hash_cache_[string_idx]; }

  const DictRef dict_ref_;
  const std::shared_ptr<StringDictionary> base_dict_;
  const int64_t base_generation_;
  size_t str_count_;
  size_t collisions_;
  std::vector<int32_t> string_id_uint32_table_;
  std::vector<uint32_t> hash_cache_;
  std::vector<int32_t> sorted_cache;
  bool materialize_hashes_;
  StringIdxEntry* offset_map_;
  char* payload_map_;
  size_t offset_file_size_;
  size_t payload_file_size_;
  size_t payload_file_off_;
  mutable mapd_shared_mutex rw_mutex_;
  mutable std::map<std::tuple<std::string, bool, bool, char>, std::vector<int32_t>>
      like_cache_;
  mutable std::map<std::pair<std::string, char>, std::vector<int32_t>> regex_cache_;
  mutable std::map<std::string, int32_t> equal_cache_;
  mutable DictionaryCache<std::string, compare_cache_value_t> compare_cache_;
  mutable std::shared_ptr<std::vector<std::string>> strings_cache_;

  char* CANARY_BUFFER{nullptr};
  size_t canary_buffer_size = 0;

  friend class ::StringDictionaryTranslator;
};

}  // namespace legacy

int32_t truncate_to_generation(const int32_t id, const size_t generation);

namespace fast {

class StringDictionary {
 public:
  StringDictionary(const DictRef& dict_ref,
                   const bool materializeHashes = false,
                   size_t initial_capacity = 256)
      : dict_ref_(dict_ref), hash_to_id_map(initial_capacity) {
    strings.reserve(initial_capacity);
  }

  int32_t getDbId() const noexcept { return dict_ref_.dbId; }
  int32_t getDictId() const noexcept { return dict_ref_.dictId; }

  class StringCallback {
   public:
    virtual ~StringCallback() = default;
    virtual void operator()(std::string const&, int32_t const string_id) = 0;
    virtual void operator()(std::string_view const, int32_t const string_id) = 0;
  };

  // Functors passed to eachStringSerially() must derive from StringCallback.
  // Each std::string const& (if isClient()) or std::string_view (if !isClient())
  // plus string_id is passed to the callback functor.
  void eachStringSerially(int64_t const generation, StringCallback&) const;
  friend class ::StringLocalCallback;

  int32_t getOrAdd(const std::string_view& str) noexcept;
  template <class T, class String>
  size_t getBulk(const std::vector<String>& string_vec, T* encoded_vec) const;
  template <class T, class String>
  size_t getBulk(const std::vector<String>& string_vec,
                 T* encoded_vec,
                 const int64_t generation) const;
  template <class T, class String>
  void getOrAddBulk(const std::vector<String>& string_vec, T* encoded_vec);
  template <class String>
  int32_t getIdOfString(const String&) const;
  std::string getString(int32_t string_id) const;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t storageEntryCount() const;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape,
                               const size_t generation) const;

  std::vector<int32_t> getCompare(const std::string& pattern,
                                  const std::string& comp_operator,
                                  const size_t generation);

  std::vector<int32_t> getRegexpLike(const std::string& pattern,
                                     const char escape,
                                     const size_t generation) const;

  std::vector<std::string> copyStrings() const;

  static constexpr int32_t INVALID_STR_ID = -1;
  static constexpr size_t MAX_STRLEN = (1 << 15) - 1;
  static constexpr size_t MAX_STRCOUNT = (1U << 31) - 1;

 private:
  int32_t getUnlocked(const std::string_view sv) const noexcept;
  std::string_view getStringFromStorageFast(const int string_id) const noexcept;
  template <class String>
  uint32_t computeBucket(
      const uint32_t hash,
      const String& input_string,
      const std::vector<int32_t>& string_id_uint32_table) const noexcept;

  template <class String>
  int32_t addString(const uint32_t hash, const String& input_string);

  const DictRef dict_ref_;
  size_t str_count_;

  struct HashMapPayload {
    int32_t string_id;
    uint32_t hash;
    std::string_view string;

    void set(const int32_t string_id_in,
             const uint32_t hash_in,
             std::string_view string_in) {
      string_id = string_id_in;
      hash = hash_in;
      string = string_in;
    }

    HashMapPayload() : string_id(INVALID_STR_ID), hash(INVALID_STR_ID) {}
  };

  std::vector<HashMapPayload> hash_to_id_map;
  std::vector<std::unique_ptr<std::string>> strings;

  // returns added string ID
  template <class String>
  int32_t addStringToMaps(const size_t bucket, const uint32_t hash, const String& str) {
    strings.emplace_back(std::make_unique<std::string>(str));
    CHECK_LT(bucket, hash_to_id_map.size());
    hash_to_id_map[bucket].set(
        static_cast<int32_t>(numStrings()) - 1, hash, *strings.back());
    return hash_to_id_map[bucket].string_id;
  }

  int32_t getIdOfStringImpl(const uint32_t hash, const std::string_view str) const;

  // returns ID for a given bucket
  int32_t id(const size_t bucket) const { return hash_to_id_map[bucket].string_id; }

  void resize(const size_t new_size);

  // returns string for a given ID
  const std::string& str(const size_t id) const { return *strings[id].get(); }

  size_t numStrings() const { return strings.size(); }

  size_t size() const { return hash_to_id_map.size(); }

  bool full() const { return strings.size() == hash_to_id_map.size(); }

  mutable mapd_shared_mutex rw_mutex_;

  // TODO: legacy, direct access outside of this class
  std::vector<uint32_t> hash_cache_;
  friend class ::StringDictionaryTranslator;
};

}  // namespace fast

#endif  // STRINGDICTIONARY_STRINGDICTIONARY_H
