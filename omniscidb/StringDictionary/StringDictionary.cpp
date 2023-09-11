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

#include "StringDictionary/StringDictionaryProxy.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <algorithm>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/sort/spreadsort/string_sort.hpp>
#include <functional>
#include <future>
#include <iostream>
#include <string_view>
#include <thread>
#include <type_traits>

// TODO(adb): fixup
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <sys/fcntl.h>
#endif

#include "Logger/Logger.h"
#include "OSDependent/omnisci_fs.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

bool g_cache_string_hash{true};

namespace {

const int SYSTEM_PAGE_SIZE = omnisci::get_page_size();

uint32_t hash_string(const std::string_view str) {
  uint32_t str_hash = 1;
  // rely on fact that unsigned overflow is defined and wraps
  for (size_t i = 0; i < str.size(); ++i) {
    str_hash = str_hash * 997 + str[i];
  }
  return str_hash;
}

template <typename T>
void fillInvalidStringIdsParallel(int64_t start, int64_t end, T* out_vec) {
  tbb::parallel_for(tbb::blocked_range<int64_t>(
                        start, end, ((size_t)256 / sizeof(T)) << 10 /* 256KB chunks*/),
                    [out_vec](const tbb::blocked_range<int64_t>& r) {
                      const int64_t start_idx = r.begin();
                      const int64_t end_idx = r.end();
                      for (int64_t string_idx = start_idx; string_idx < end_idx;
                           ++string_idx) {
                        out_vec[string_idx] = StringDictionary::INVALID_STR_ID;
                      }
                    });
}

}  // namespace

bool g_enable_stringdict_parallel{false};

namespace legacy {

constexpr int32_t StringDictionary::INVALID_STR_ID;
constexpr size_t StringDictionary::MAX_STRLEN;
constexpr size_t StringDictionary::MAX_STRCOUNT;

StringDictionary::StringDictionary(const DictRef& dict_ref,
                                   const bool materializeHashes,
                                   size_t initial_capacity)
    : dict_ref_(dict_ref)
    , base_generation_(0)
    , str_count_(0)
    // Search code assumes non-empty table.
    , string_id_uint32_table_(std::max(initial_capacity, (size_t)2), INVALID_STR_ID)
    , hash_cache_(std::max(initial_capacity, (size_t)2))
    , materialize_hashes_(materializeHashes)
    , offset_map_(nullptr)
    , payload_map_(nullptr)
    , offset_file_size_(0)
    , payload_file_size_(0)
    , payload_file_off_(0)
    , strings_cache_(nullptr) {
  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(size_t(0), (initial_capacity & (initial_capacity - 1)));
}

StringDictionary::StringDictionary(std::shared_ptr<StringDictionary> base_dict,
                                   const int64_t generation,
                                   const bool materializeHashes,
                                   size_t initial_capacity)
    : dict_ref_(-1, -1)
    , base_dict_(base_dict)
    , base_generation_(generation >= 0 ? generation
                                       : static_cast<int64_t>(base_dict->entryCount()))
    , str_count_(0)
    // Search code assumes non-empty table.
    , string_id_uint32_table_(std::max(initial_capacity, (size_t)2), INVALID_STR_ID)
    , hash_cache_(std::max(initial_capacity, (size_t)2))
    , materialize_hashes_(materializeHashes)
    , offset_map_(nullptr)
    , payload_map_(nullptr)
    , offset_file_size_(0)
    , payload_file_size_(0)
    , payload_file_off_(0)
    , strings_cache_(nullptr) {}

namespace {
class MapMaker : public StringDictionary::StringCallback {
  std::unordered_map<std::string, int32_t> map_;

 public:
  void operator()(std::string const& str, int32_t const string_id) override {
    auto const emplaced = map_.emplace(str, string_id);
    CHECK(emplaced.second) << "str(" << str << ") string_id(" << string_id << ')';
  }
  void operator()(std::string_view const, int32_t const string_id) override {
    UNREACHABLE() << "MapMaker must be called with a std::string.";
  }
  std::unordered_map<std::string, int32_t> moveMap() { return std::move(map_); }
};
}  // namespace

// Call serial_callback for each (string/_view, string_id). Must be called serially.
void StringDictionary::eachStringSerially(int64_t const generation,
                                          StringCallback& serial_callback) const {
  if (base_dict_) {
    auto generation_for_base =
        generation >= 0 ? std::min(generation, base_generation_) : base_generation_;
    base_dict_->eachStringSerially(generation_for_base, serial_callback);
  }
  size_t const n = std::min(static_cast<size_t>(generation), entryCount());
  CHECK_LE(n, static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1);
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  for (unsigned id = base_generation_; id < n; ++id) {
    serial_callback(getStringFromStorageFast(static_cast<int>(id)), id);
  }
}

void StringDictionary::eachStringSerially(StringCallback& serial_callback) const {
  eachStringSerially(-1, serial_callback);
}

int32_t StringDictionary::getDbId() const noexcept {
  return dict_ref_.dbId;
}

int32_t StringDictionary::getDictId() const noexcept {
  return dict_ref_.dictId;
}

StringDictionary::~StringDictionary() noexcept {
  free(CANARY_BUFFER);
  if (payload_map_) {
    CHECK(offset_map_);
    free(payload_map_);
    free(offset_map_);
  }
}

int32_t StringDictionary::getOrAdd(const std::string_view& str) noexcept {
  // @TODO(wei) treat empty string as NULL for now
  if (str.size() == 0) {
    return inline_int_null_value<int32_t>();
  }
  CHECK(str.size() <= MAX_STRLEN);
  const uint32_t hash = hash_string(str);
  if (base_dict_) {
    auto base_res = base_dict_->getIdOfString(str, hash);
    if (base_res != INVALID_STR_ID && base_res < base_generation_) {
      return base_res;
    }
  }
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
    const uint32_t bucket = computeBucket(hash, str, string_id_uint32_table_);
    if (string_id_uint32_table_[bucket] != INVALID_STR_ID) {
      return string_id_uint32_table_[bucket];
    }
  }
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (fillRateIsHigh(str_count_)) {
    // resize when more than 50% is full
    increaseHashTableCapacity();
  }
  // need to recalculate the bucket in case it changed before
  // we got the lock
  const uint32_t bucket = computeBucket(hash, str, string_id_uint32_table_);
  if (string_id_uint32_table_[bucket] == INVALID_STR_ID) {
    CHECK_LT(str_count_, MAX_STRCOUNT)
        << "Maximum number (" << str_count_
        << ") of Dictionary encoded Strings reached for this column";
    appendToStorage(str);
    string_id_uint32_table_[bucket] = indexToId(str_count_);
    if (materialize_hashes_) {
      hash_cache_[str_count_] = hash;
    }
    ++str_count_;
    invalidateInvertedIndex();
  }
  return string_id_uint32_table_[bucket];
}

namespace {

template <class T>
void throw_encoding_error(std::string_view str, const DictRef& dict_ref) {
  std::ostringstream oss;
  oss << "The text encoded column using dictionary " << dict_ref.toString()
      << " has exceeded it's limit of " << sizeof(T) * 8 << " bits ("
      << static_cast<size_t>(max_valid_int_value<T>() + 1) << " unique values) "
      << "while attempting to add the new string '" << str << "'. ";

  if (sizeof(T) < 4) {
    // Todo: Implement automatic type widening for dictionary-encoded text
    // columns/all fixed length columm types (at least if not defined
    //  with fixed encoding size), or short of that, ALTER TABLE
    // COLUMN TYPE to at least allow the user to do this manually
    // without re-creating the table

    oss << "To load more data, please re-create the table with "
        << "this column as type TEXT ENCODING DICT(" << sizeof(T) * 2 * 8 << ") ";
    if (sizeof(T) == 1) {
      oss << "or TEXT ENCODING DICT(32) ";
    }
    oss << "and reload your data.";
  } else {
    // Todo: Implement TEXT ENCODING DICT(64) type which should essentially
    // preclude overflows.
    oss << "Currently dictionary-encoded text columns support a maximum of "
        << StringDictionary::MAX_STRCOUNT
        << " strings. Consider recreating the table with "
        << "this column as type TEXT ENCODING NONE and reloading your data.";
  }
  LOG(ERROR) << oss.str();
  throw std::runtime_error(oss.str());
}

void throw_string_too_long_error(std::string_view str, const DictRef& dict_ref) {
  std::ostringstream oss;
  oss << "The string '" << str << " could not be inserted into the dictionary "
      << dict_ref.toString() << " because it exceeded the maximum allowable "
      << "length of " << StringDictionary::MAX_STRLEN << " characters (string was "
      << str.size() << " characters).";
  LOG(ERROR) << oss.str();
  throw std::runtime_error(oss.str());
}

}  // namespace

/**
 * Method to hash a vector of strings in parallel.
 * @param string_vec input vector of strings to be hashed
 * @param hashes space for the output - should be pre-sized to match string_vec size
 */
template <class String>
void StringDictionary::hashStrings(const std::vector<String>& string_vec,
                                   std::vector<uint32_t>& hashes) const noexcept {
  CHECK_EQ(string_vec.size(), hashes.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, string_vec.size()),
                    [&string_vec, &hashes](const tbb::blocked_range<size_t>& r) {
                      for (size_t curr_id = r.begin(); curr_id != r.end(); ++curr_id) {
                        if (string_vec[curr_id].empty()) {
                          continue;
                        }
                        hashes[curr_id] = hash_string(string_vec[curr_id]);
                      }
                    });
}

template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec) const {
  return getBulk(string_vec, encoded_vec, -1L /* generation */);
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec) const;

template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec,
                                 const int64_t generation) const {
  constexpr int64_t target_strings_per_thread{1000};
  const int64_t num_lookup_strings = string_vec.size();
  if (num_lookup_strings == 0) {
    return 0;
  }

  size_t base_num_strings_not_found = string_vec.size();
  if (base_dict_) {
    auto base_generation_for_bulk =
        generation >= 0 ? std::min(generation, base_generation_) : base_generation_;
    base_num_strings_not_found =
        base_dict_->getBulk(string_vec, encoded_vec, base_generation_for_bulk);
  }

  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  const int64_t num_dict_strings = generation >= 0 ? generation : entryCount();
  const bool skip_owned_string =
      (num_dict_strings <= base_generation_) || !base_num_strings_not_found;
  if (skip_owned_string) {
    // Need to fill the resulting vector if it wasn't done by the base dictionary.
    if (!base_dict_) {
      fillInvalidStringIdsParallel(0, num_lookup_strings, encoded_vec);
    }
    return base_num_strings_not_found;
  }
  // If we're here the generation-capped dictionary has strings in it
  // that we need to look up against
  size_t found_owned = tbb::parallel_reduce(
      tbb::blocked_range<int64_t>(
          0, num_lookup_strings, target_strings_per_thread /* tbb grain_size */),
      (size_t)0,
      [&](const tbb::blocked_range<int64_t>& r, size_t found) {
        const int64_t start_idx = r.begin();
        const int64_t end_idx = r.end();
        for (int64_t string_idx = start_idx; string_idx != end_idx; ++string_idx) {
          if (base_dict_ && encoded_vec[string_idx] != StringDictionary::INVALID_STR_ID) {
            continue;
          }
          const auto& input_string = string_vec[string_idx];
          if (input_string.empty()) {
            encoded_vec[string_idx] = inline_int_null_value<T>();
            ++found;
            continue;
          }
          if (input_string.size() > StringDictionary::MAX_STRLEN) {
            throw_string_too_long_error(input_string, dict_ref_);
          }
          // Will either be legit id or INVALID_STR_ID
          const auto string_id = getOwnedUnlocked(input_string);
          if (string_id == StringDictionary::INVALID_STR_ID ||
              string_id >= num_dict_strings) {
            encoded_vec[string_idx] = StringDictionary::INVALID_STR_ID;
            continue;
          }
          encoded_vec[string_idx] = string_id;
          ++found;
        }
        return found;
      },
      std::plus<size_t>());

  return base_num_strings_not_found - found_owned;
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec,
                                          const int64_t generation) const;

template <class T, class String>
void StringDictionary::getOrAddBulk(const std::vector<String>& input_strings,
                                    T* output_string_ids) {
  if (base_dict_) {
    auto missing_count =
        base_dict_->getBulk(input_strings, output_string_ids, base_generation_);
    if (!missing_count) {
      return;
    }
  }

  if (g_enable_stringdict_parallel) {
    getOrAddBulkParallel(input_strings, output_string_ids);
    return;
  }
  // Single-thread path.
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);

  const size_t initial_str_count = str_count_;
  size_t idx = 0;
  for (const auto& input_string : input_strings) {
    // Skip strings found in the base dictionary.
    if (base_dict_ && output_string_ids[idx] != INVALID_STR_ID) {
      ++idx;
      continue;
    }

    if (input_string.empty()) {
      output_string_ids[idx++] = inline_int_null_value<T>();
      continue;
    }
    CHECK(input_string.size() <= MAX_STRLEN);

    const uint32_t input_string_hash = hash_string(input_string);
    uint32_t hash_bucket =
        computeBucket(input_string_hash, input_string, string_id_uint32_table_);
    if (string_id_uint32_table_[hash_bucket] != INVALID_STR_ID) {
      output_string_ids[idx++] = string_id_uint32_table_[hash_bucket];
      continue;
    }
    // need to add record to dictionary
    // check there is room
    if (str_count_ > static_cast<size_t>(max_valid_int_value<T>())) {
      throw_encoding_error<T>(input_string, dict_ref_);
    }
    CHECK_LT(str_count_, MAX_STRCOUNT)
        << "Maximum number (" << str_count_
        << ") of Dictionary encoded Strings reached for this column";
    if (fillRateIsHigh(str_count_)) {
      // resize when more than 50% is full
      increaseHashTableCapacity();
      hash_bucket =
          computeBucket(input_string_hash, input_string, string_id_uint32_table_);
    }
    appendToStorage(input_string);

    if (materialize_hashes_) {
      hash_cache_[str_count_] = input_string_hash;
    }
    const int32_t string_id = indexToId(str_count_);
    string_id_uint32_table_[hash_bucket] = string_id;
    output_string_ids[idx++] = string_id;
    ++str_count_;
  }
  const size_t num_strings_added = str_count_ - initial_str_count;
  if (num_strings_added > 0) {
    invalidateInvertedIndex();
  }
}

template <class T, class String>
void StringDictionary::getOrAddBulkParallel(const std::vector<String>& input_strings,
                                            T* output_string_ids) {
  // Compute hashes of the input strings up front, and in parallel,
  // as the string hashing does not need to be behind the subsequent write_lock
  std::vector<uint32_t> input_strings_hashes(input_strings.size());
  hashStrings(input_strings, input_strings_hashes);

  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  size_t shadow_str_count =
      str_count_;  // Need to shadow str_count_ now with bulk add methods
  const size_t storage_high_water_mark = shadow_str_count;
  std::vector<size_t> string_memory_ids;
  size_t sum_new_string_lengths = 0;
  string_memory_ids.reserve(input_strings.size());
  size_t input_string_idx{0};
  for (const auto& input_string : input_strings) {
    // Skip strings found in the base dictionary.
    if (base_dict_ && output_string_ids[input_string_idx] != INVALID_STR_ID) {
      ++input_string_idx;
      continue;
    }

    // Currently we make empty strings null
    if (input_string.empty()) {
      output_string_ids[input_string_idx++] = inline_int_null_value<T>();
      continue;
    }
    // TODO: Recover gracefully if an input string is too long
    CHECK(input_string.size() <= MAX_STRLEN);

    if (fillRateIsHigh(shadow_str_count)) {
      // resize when more than 50% is full
      increaseHashTableCapacityFromStorageAndMemory(shadow_str_count,
                                                    storage_high_water_mark,
                                                    input_strings,
                                                    string_memory_ids,
                                                    input_strings_hashes);
    }
    // Compute the hash for this input_string
    const uint32_t input_string_hash = input_strings_hashes[input_string_idx];

    const uint32_t hash_bucket =
        computeBucketFromStorageAndMemory(input_string_hash,
                                          input_string,
                                          string_id_uint32_table_,
                                          storage_high_water_mark,
                                          input_strings,
                                          string_memory_ids);

    // If the hash bucket is not empty, that is our string id
    // (computeBucketFromStorageAndMemory) already checked to ensure the input string and
    // bucket string are equal)
    if (string_id_uint32_table_[hash_bucket] != INVALID_STR_ID) {
      output_string_ids[input_string_idx++] = string_id_uint32_table_[hash_bucket];
      continue;
    }
    // Did not find string, so need to add record to dictionary
    // First check there is room
    if (shadow_str_count > static_cast<size_t>(max_valid_int_value<T>())) {
      throw_encoding_error<T>(input_string, dict_ref_);
    }
    CHECK_LT(shadow_str_count, MAX_STRCOUNT)
        << "Maximum number (" << shadow_str_count
        << ") of Dictionary encoded Strings reached for this column";
    string_memory_ids.push_back(input_string_idx);
    sum_new_string_lengths += input_string.size();
    string_id_uint32_table_[hash_bucket] = indexToId(shadow_str_count);
    if (materialize_hashes_) {
      hash_cache_[shadow_str_count] = input_string_hash;
    }
    output_string_ids[input_string_idx++] = indexToId(shadow_str_count++);
  }
  appendToStorageBulk(input_strings, string_memory_ids, sum_new_string_lengths);
  const size_t num_strings_added = shadow_str_count - str_count_;
  str_count_ = shadow_str_count;
  if (num_strings_added > 0) {
    invalidateInvertedIndex();
  }
}
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             int32_t* encoded_vec);

template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    int32_t* encoded_vec);

template <class String>
std::vector<int32_t> StringDictionary::getOrAddBulk(
    const std::vector<String>& string_vec) {
  std::vector<int32_t> res(string_vec.size());
  getOrAddBulk(string_vec, res.data());
  return res;
}

template std::vector<int32_t> StringDictionary::getOrAddBulk(
    const std::vector<std::string>& string_vec);
template std::vector<int32_t> StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec);

template <class String>
std::vector<int32_t> StringDictionary::getBulk(const std::vector<String>& string_vec) {
  std::vector<int32_t> res(string_vec.size());
  getBulk(string_vec, res.data());
  return res;
}

template std::vector<int32_t> StringDictionary::getBulk(
    const std::vector<std::string>& string_vec);
template std::vector<int32_t> StringDictionary::getBulk(
    const std::vector<std::string_view>& string_vec);

template <class String>
int32_t StringDictionary::getIdOfString(const String& str) const {
  return getIdOfString(str, hash_string(str));
}

template int32_t StringDictionary::getIdOfString(const std::string&) const;
template int32_t StringDictionary::getIdOfString(const std::string_view&) const;

template <class String>
int32_t StringDictionary::getIdOfString(const String& str, const uint32_t hash) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getUnlocked(str, hash);
}

template int32_t StringDictionary::getIdOfString(const std::string&,
                                                 const uint32_t) const;
template int32_t StringDictionary::getIdOfString(const std::string_view&,
                                                 const uint32_t) const;

int32_t StringDictionary::getUnlocked(const std::string_view sv) const noexcept {
  return getUnlocked(sv, hash_string(sv));
}

int32_t StringDictionary::getUnlocked(const std::string_view sv,
                                      const uint32_t hash) const noexcept {
  if (base_dict_) {
    auto base_res = base_dict_->getIdOfString(sv, hash);
    if (base_res != INVALID_STR_ID && base_res < base_generation_) {
      return base_res;
    }
  }
  return getOwnedUnlocked(sv, hash);
}

int32_t StringDictionary::getOwnedUnlocked(const std::string_view sv) const noexcept {
  return getOwnedUnlocked(sv, hash_string(sv));
}

int32_t StringDictionary::getOwnedUnlocked(const std::string_view sv,
                                           const uint32_t hash) const noexcept {
  auto str_id = string_id_uint32_table_[computeBucket(hash, sv, string_id_uint32_table_)];
  return str_id;
}

std::string StringDictionary::getString(int32_t string_id) const {
  if (inline_int_null_value<int32_t>() == string_id) {
    return "";
  }
  if (string_id < base_generation_) {
    return base_dict_->getString(string_id);
  }
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getOwnedStringChecked(string_id);
}

std::string StringDictionary::getStringUnlocked(int32_t string_id) const noexcept {
  if (string_id < base_generation_) {
    return base_dict_->getString(string_id);
  }
  return getOwnedStringChecked(string_id);
}

std::pair<char*, size_t> StringDictionary::getStringBytes(
    int32_t string_id) const noexcept {
  if (string_id < base_generation_) {
    return base_dict_->getStringBytes(string_id);
  }
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK_LE(0, string_id);
  return getOwnedStringBytesChecked(string_id);
}

size_t StringDictionary::storageEntryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return str_count_;
}

size_t StringDictionary::entryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return str_count_ + base_generation_;
}

namespace {

bool is_like(const std::string& str,
             const std::string& pattern,
             const bool icase,
             const bool is_simple,
             const char escape) {
  return icase
             ? (is_simple ? string_ilike_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_ilike(str.c_str(),
                                         str.size(),
                                         pattern.c_str(),
                                         pattern.size(),
                                         escape))
             : (is_simple ? string_like_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_like(str.c_str(),
                                        str.size(),
                                        pattern.c_str(),
                                        pattern.size(),
                                        escape));
}

}  // namespace

std::vector<int32_t> StringDictionary::getLike(const std::string& pattern,
                                               const bool icase,
                                               const bool is_simple,
                                               const char escape,
                                               int64_t generation) const {
  generation = generation >= 0 ? std::min(generation, static_cast<int64_t>(entryCount()))
                               : static_cast<int64_t>(entryCount());
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  const auto cache_key = std::make_tuple(pattern, icase, is_simple, escape, generation);
  const auto it = like_cache_.find(cache_key);
  if (it != like_cache_.end()) {
    return it->second;
  }

  std::vector<int32_t> result;
  if (base_dict_) {
    result = base_dict_->getLike(
        pattern, icase, is_simple, escape, std::min(generation, base_generation_));
  }

  if (generation < base_generation_) {
    return result;
  }

  std::vector<std::thread> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<int32_t>> worker_results(worker_count);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&worker_results,
                          &pattern,
                          generation,
                          icase,
                          is_simple,
                          escape,
                          worker_idx,
                          worker_count,
                          this]() {
      for (int string_id = indexToId(worker_idx); string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_like(str, pattern, icase, is_simple, escape)) {
          worker_results[worker_idx].push_back(string_id);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& worker_result : worker_results) {
    result.insert(result.end(), worker_result.begin(), worker_result.end());
  }
  // place result into cache for reuse if similar query
  const auto it_ok = like_cache_.insert(std::make_pair(cache_key, result));

  CHECK(it_ok.second);

  return result;
}

std::vector<int32_t> StringDictionary::getEquals(const std::string& pattern,
                                                 const std::string& comp_operator,
                                                 int64_t generation) const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  std::vector<int32_t> result;
  if (base_dict_) {
    result = base_dict_->getEquals(
        pattern, comp_operator, std::min(generation, base_generation_));
    if ((comp_operator == "=" && !result.empty()) || generation < base_generation_) {
      return result;
    }
  }

  auto eq_id_itr = equal_cache_.find(pattern);
  int32_t eq_id = -1;
  if (eq_id_itr != equal_cache_.end()) {
    auto eq_id = eq_id_itr->second;
    if (comp_operator == "=") {
      if (eq_id < generation) {
        result.push_back(eq_id);
      }
    } else {
      for (int32_t id = base_generation_; id < generation; id++) {
        if (id != eq_id) {
          result.push_back(id);
        }
      }
    }
  } else {
    std::vector<std::thread> workers;
    int worker_count = cpu_threads();
    CHECK_GT(worker_count, 0);
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back(
          [&eq_id, &pattern, generation, worker_idx, worker_count, this]() {
            for (int string_id = indexToId(worker_idx); string_id < generation;
                 string_id += worker_count) {
              const auto str = getStringUnlocked(string_id);
              if (str == pattern) {
                // Only one thread can find matching string, so no additional sync.
                eq_id = string_id;
                break;
              }
            }
          });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    if (eq_id >= 0) {
      const auto it_ok = equal_cache_.insert(std::make_pair(pattern, eq_id));
      CHECK(it_ok.second);
    }
    if (comp_operator == "<>") {
      for (int32_t id = base_generation_; id < generation; id++) {
        if (id != eq_id) {
          result.push_back(id);
        }
      }
    } else if (eq_id >= 0 && eq_id < generation) {
      result.push_back(eq_id);
    }
  }
  return result;
}

std::vector<int32_t> StringDictionary::getCompare(const std::string& pattern,
                                                  const std::string& comp_operator,
                                                  int64_t generation) const {
  generation = generation >= 0 ? std::min(generation, static_cast<int64_t>(entryCount()))
                               : static_cast<int64_t>(entryCount());
  {
    // The lock is used only to check cache.
    mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
    if ((sorted_cache.size() < str_count_) &&
        (comp_operator == "=" || comp_operator == "<>")) {
      read_lock.unlock();
      return getEquals(pattern, comp_operator, generation);
    }
  }

  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  std::vector<int32_t> ret;
  if (base_dict_) {
    ret = base_dict_->getCompare(
        pattern, comp_operator, std::min(generation, base_generation_));
    if ((comp_operator == "=" && !ret.empty()) || generation < base_generation_) {
      return ret;
    }
  }

  if (sorted_cache.size() < str_count_) {
    buildSortedCache();
  }

  auto cache_index = compare_cache_.get(pattern);

  if (!cache_index) {
    cache_index = std::make_shared<StringDictionary::compare_cache_value_t>();
    const auto cache_itr = std::lower_bound(
        sorted_cache.begin(),
        sorted_cache.end(),
        pattern,
        [this](decltype(sorted_cache)::value_type const& a, decltype(pattern)& b) {
          auto a_str = this->getStringFromStorage(a);
          return string_lt(a_str.c_str_ptr, a_str.size, b.c_str(), b.size());
        });

    if (cache_itr == sorted_cache.end()) {
      cache_index->index = sorted_cache.size() - 1;
      cache_index->diff = 1;
    } else {
      const auto cache_str = getStringFromStorage(*cache_itr);
      if (!string_eq(
              cache_str.c_str_ptr, cache_str.size, pattern.c_str(), pattern.size())) {
        cache_index->index = cache_itr - sorted_cache.begin() - 1;
        cache_index->diff = 1;
      } else {
        cache_index->index = cache_itr - sorted_cache.begin();
        cache_index->diff = 0;
      }
    }

    compare_cache_.put(pattern, cache_index);
  }

  // since we have a cache in form of vector of ints which is sorted according to
  // corresponding strings in the dictionary all we need is the index of the element
  // which equal to the pattern that we are trying to match or the index of "biggest"
  // element smaller than the pattern, to perform all the comparison operators over
  // string. The search function guarantees we have such index so now it is just the
  // matter to include all the elements in the result vector.

  // For < operator if the index that we have points to the element which is equal to
  // the pattern that we are searching for we simply get all the elements less than the
  // index. If the element pointed by the index is not equal to the pattern we are
  // comparing with we also need to include that index in result vector.

  if (comp_operator == "<") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
    }
    for (size_t i = 0; i < idx; i++) {
      if (sorted_cache[i] < generation) {
        ret.push_back(sorted_cache[i]);
      }
    }

    // For <= operator we want to include the all elements less than the index and
    // the index itself since it cannot be greater than the pattern.
  } else if (comp_operator == "<=") {
    size_t idx = cache_index->index + 1;
    for (size_t i = 0; i < idx; i++) {
      if (sorted_cache[i] < generation) {
        ret.push_back(sorted_cache[i]);
      }
    }

    // For > operator we want to get all the elements with index greater than the index
  } else if (comp_operator == ">") {
    size_t idx = cache_index->index + 1;
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      if (sorted_cache[i] < generation) {
        ret.push_back(sorted_cache[i]);
      }
    }

    // For >= operator we want to get all the elements with index greater than the index.
    // We also include the index if it matches the pattern
  } else if (comp_operator == ">=") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
    }
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      if (sorted_cache[i] < generation) {
        ret.push_back(sorted_cache[i]);
      }
    }
  } else if (comp_operator == "=") {
    if (!cache_index->diff) {
      if (sorted_cache[cache_index->index] < generation) {
        ret.push_back(sorted_cache[cache_index->index]);
      }
    }

    // For <> operator it is simple matter of not including id of string which is equal
    // to pattern we are searching for.
  } else if (comp_operator == "<>") {
    if (!cache_index->diff) {
      int eq_id = sorted_cache[cache_index->index];
      for (int id = base_generation_; id < generation; ++id) {
        if (id != eq_id) {
          ret.push_back(id);
        }
      }
    } else {
      for (int id = base_generation_; id < generation; ++id) {
        ret.push_back(id);
      }
    }
  } else {
    std::runtime_error("Unsupported string comparison operator");
  }
  return ret;
}

namespace {

bool is_regexp_like(const std::string& str,
                    const std::string& pattern,
                    const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<int32_t> StringDictionary::getRegexpLike(const std::string& pattern,
                                                     const char escape,
                                                     int64_t generation) const {
  generation = generation >= 0 ? std::min(generation, static_cast<int64_t>(entryCount()))
                               : static_cast<int64_t>(entryCount());

  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  const auto cache_key = std::make_tuple(pattern, escape, generation);
  const auto it = regex_cache_.find(cache_key);
  if (it != regex_cache_.end()) {
    return it->second;
  }

  std::vector<int32_t> result;
  if (base_dict_) {
    result = base_dict_->getRegexpLike(
        pattern, escape, std::min(generation, base_generation_));
    if (generation < base_generation_) {
      return result;
    }
  }

  std::vector<std::thread> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<int32_t>> worker_results(worker_count);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&worker_results,
                          &pattern,
                          generation,
                          escape,
                          worker_idx,
                          worker_count,
                          this]() {
      for (int string_id = indexToId(worker_idx); string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_regexp_like(str, pattern, escape)) {
          worker_results[worker_idx].push_back(string_id);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& worker_result : worker_results) {
    result.insert(result.end(), worker_result.begin(), worker_result.end());
  }
  const auto it_ok = regex_cache_.insert(std::make_pair(cache_key, result));
  CHECK(it_ok.second);

  return result;
}

std::vector<std::string> StringDictionary::copyStrings(int64_t generation) const {
  generation = generation >= 0 ? std::min(generation, static_cast<int64_t>(entryCount()))
                               : static_cast<int64_t>(entryCount());
  if (!strings_cache_) {
    strings_cache_ = std::make_shared<std::vector<std::string>>();
    strings_cache_->reserve(entryCount());
    copyStrings(0, entryCount(), *strings_cache_);
  } else if (strings_cache_->size() < static_cast<size_t>(generation)) {
    auto start = strings_cache_->size();
    strings_cache_->reserve(entryCount());
    copyStrings(start, entryCount(), *strings_cache_);
  }

  return std::vector<std::string>(strings_cache_->begin(),
                                  strings_cache_->begin() + generation);
}

void StringDictionary::copyStrings(int64_t string_id_start,
                                   int64_t string_id_end,
                                   std::vector<std::string>& out_vec) const {
  CHECK_GE(string_id_start, 0);
  CHECK_LE(string_id_end, static_cast<int64_t>(entryCount()));

  if (base_dict_ && string_id_start < base_generation_) {
    base_dict_->copyStrings(
        string_id_start, std::min(base_generation_, string_id_end), out_vec);
  }

  int64_t local_string_id_start = std::max(string_id_start, base_generation_);
  int64_t local_string_id_end = string_id_end;
  if (local_string_id_start >= local_string_id_end) {
    return;
  }

  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  const bool multithreaded = (local_string_id_end - local_string_id_start) > 10000;
  auto copy = [this](std::vector<std::string>& str_list,
                     const int64_t start_id,
                     const int64_t end_id) {
    CHECK_LE(start_id, end_id);
    str_list.reserve(end_id - start_id);
    for (int64_t string_id = start_id; string_id < end_id; ++string_id) {
      str_list.push_back(getStringUnlocked(string_id));
    }
  };
  if (multithreaded) {
    const auto worker_count = cpu_threads();
    CHECK_GT(worker_count, 0);
    std::vector<std::vector<std::string>> worker_results(worker_count);
    std::vector<std::future<void>> workers;
    const auto stride =
        (local_string_id_end - local_string_id_start + (worker_count - 1)) / worker_count;
    for (int64_t worker_idx = 0,
                 start = local_string_id_start,
                 end = std::min(start + stride, local_string_id_end);
         worker_idx < worker_count && start < local_string_id_end;
         ++worker_idx,
                 start += stride,
                 end = std::min(start + stride, local_string_id_end)) {
      workers.push_back(std::async(
          std::launch::async, copy, std::ref(worker_results[worker_idx]), start, end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
    for (const auto& worker_result : worker_results) {
      out_vec.insert(out_vec.end(), worker_result.begin(), worker_result.end());
    }
  } else {
    copy(out_vec, local_string_id_start, local_string_id_end);
  }
}

bool StringDictionary::fillRateIsHigh(const size_t num_strings) const noexcept {
  return string_id_uint32_table_.size() <= num_strings * 2;
}

void StringDictionary::increaseHashTableCapacity() noexcept {
  std::vector<int32_t> new_str_ids(string_id_uint32_table_.size() * 2, INVALID_STR_ID);

  if (materialize_hashes_) {
    for (size_t i = 0; i != str_count_; ++i) {
      const uint32_t hash = hash_cache_[i];
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = indexToId(i);
    }
    hash_cache_.resize(hash_cache_.size() * 2);
  } else {
    for (size_t i = 0; i != str_count_; ++i) {
      const auto str = getOwnedStringChecked(indexToId(i));
      const uint32_t hash = hash_string(str);
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = indexToId(i);
    }
  }
  string_id_uint32_table_.swap(new_str_ids);
}

template <class String>
void StringDictionary::increaseHashTableCapacityFromStorageAndMemory(
    const size_t str_count,  // str_count_ is only persisted strings, so need transient
                             // shadow count
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const std::vector<uint32_t>& input_strings_hashes) noexcept {
  std::vector<int32_t> new_str_ids(string_id_uint32_table_.size() * 2, INVALID_STR_ID);
  if (materialize_hashes_) {
    for (size_t i = 0; i != str_count; ++i) {
      const uint32_t hash = hash_cache_[i];
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = indexToId(i);
    }
    hash_cache_.resize(hash_cache_.size() * 2);
  } else {
    for (size_t storage_idx = 0; storage_idx != storage_high_water_mark; ++storage_idx) {
      const auto storage_string = getOwnedStringChecked(indexToId(storage_idx));
      const uint32_t hash = hash_string(storage_string);
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = indexToId(storage_idx);
    }
    for (size_t memory_idx = 0; memory_idx != string_memory_ids.size(); ++memory_idx) {
      const size_t string_memory_id = string_memory_ids[memory_idx];
      const uint32_t bucket = computeUniqueBucketWithHash(
          input_strings_hashes[string_memory_id], new_str_ids);
      new_str_ids[bucket] = indexToId(storage_high_water_mark + memory_idx);
    }
  }
  string_id_uint32_table_.swap(new_str_ids);
}

std::string StringDictionary::getOwnedStringChecked(const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::string(str_canary.c_str_ptr, str_canary.size);
}

std::pair<char*, size_t> StringDictionary::getOwnedStringBytesChecked(
    const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::make_pair(str_canary.c_str_ptr, str_canary.size);
}

template <class String>
uint32_t StringDictionary::computeBucket(
    const uint32_t hash,
    const String& input_string,
    const std::vector<int32_t>& string_id_uint32_table) const noexcept {
  const size_t uint32_table_size = string_id_uint32_table.size();
  uint32_t bucket = hash & (uint32_table_size - 1);
  while (true) {
    const int32_t candidate_string_id = string_id_uint32_table[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    if ((materialize_hashes_ && hash == hashById(candidate_string_id)) ||
        !materialize_hashes_) {
      const auto candidate_string = getStringFromStorageFast(candidate_string_id);
      if (input_string.size() == candidate_string.size() &&
          !memcmp(input_string.data(), candidate_string.data(), input_string.size())) {
        // found the string
        break;
      }
    }
    // wrap around
    if (++bucket == uint32_table_size) {
      bucket = 0;
    }
  }
  return bucket;
}

template <class String>
uint32_t StringDictionary::computeBucketFromStorageAndMemory(
    const uint32_t input_string_hash,
    const String& input_string,
    const std::vector<int32_t>& string_id_uint32_table,
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids) const noexcept {
  uint32_t bucket = input_string_hash & (string_id_uint32_table.size() - 1);
  while (true) {
    const int32_t candidate_string_id = string_id_uint32_table[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    auto candidate_string_index = idToIndex(candidate_string_id);
    if (!materialize_hashes_ || (input_string_hash == hashById(candidate_string_id))) {
      if (candidate_string_index > 0 &&
          static_cast<size_t>(candidate_string_index) >= storage_high_water_mark) {
        // The candidate string is not in storage yet but in our string_memory_ids temp
        // buffer
        size_t memory_offset =
            static_cast<size_t>(candidate_string_index - storage_high_water_mark);
        const String candidate_string = input_strings[string_memory_ids[memory_offset]];
        if (input_string.size() == candidate_string.size() &&
            !memcmp(input_string.data(), candidate_string.data(), input_string.size())) {
          // found the string in the temp memory buffer
          break;
        }
      } else {
        // The candidate string is in storage, need to fetch it for comparison
        const auto candidate_storage_string =
            getStringFromStorageFast(candidate_string_id);
        if (input_string.size() == candidate_storage_string.size() &&
            !memcmp(input_string.data(),
                    candidate_storage_string.data(),
                    input_string.size())) {
          //! memcmp(input_string.data(), candidate_storage_string.c_str_ptr,
          //! input_string.size())) {
          // found the string in storage
          break;
        }
      }
    }
    if (++bucket == string_id_uint32_table.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

uint32_t StringDictionary::computeUniqueBucketWithHash(
    const uint32_t hash,
    const std::vector<int32_t>& string_id_uint32_table) noexcept {
  const size_t uint32_table_size = string_id_uint32_table.size();
  uint32_t bucket = hash & (uint32_table_size - 1);
  while (true) {
    if (string_id_uint32_table[bucket] ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    collisions_++;
    // wrap around
    if (++bucket == uint32_table_size) {
      bucket = 0;
    }
  }
  return bucket;
}

void StringDictionary::checkAndConditionallyIncreasePayloadCapacity(
    const size_t write_length) {
  if (payload_file_off_ + write_length > payload_file_size_) {
    const size_t min_capacity_needed =
        write_length - (payload_file_size_ - payload_file_off_);
    addPayloadCapacity(min_capacity_needed);
    CHECK(payload_file_off_ + write_length <= payload_file_size_);
  }
}

void StringDictionary::checkAndConditionallyIncreaseOffsetCapacity(
    const size_t write_length) {
  const size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  if (offset_file_off + write_length >= offset_file_size_) {
    const size_t min_capacity_needed =
        write_length - (offset_file_size_ - offset_file_off);
    addOffsetCapacity(min_capacity_needed);
    CHECK(offset_file_off + write_length <= offset_file_size_);
  }
}

template <class String>
void StringDictionary::appendToStorage(const String str) noexcept {
  // write the payload
  checkAndConditionallyIncreasePayloadCapacity(str.size());
  memcpy(payload_map_ + payload_file_off_, str.data(), str.size());

  // write the offset and length
  StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str.size()};
  payload_file_off_ += str.size();  // Need to increment after we've defined str_meta

  checkAndConditionallyIncreaseOffsetCapacity(sizeof(str_meta));
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

template <class String>
void StringDictionary::appendToStorageBulk(
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const size_t sum_new_strings_lengths) noexcept {
  const size_t num_strings = string_memory_ids.size();

  checkAndConditionallyIncreasePayloadCapacity(sum_new_strings_lengths);
  checkAndConditionallyIncreaseOffsetCapacity(sizeof(StringIdxEntry) * num_strings);

  for (size_t i = 0; i < num_strings; ++i) {
    const size_t string_idx = string_memory_ids[i];
    const String str = input_strings[string_idx];
    const size_t str_size(str.size());
    memcpy(payload_map_ + payload_file_off_, str.data(), str_size);
    StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str_size};
    payload_file_off_ += str_size;  // Need to increment after we've defined str_meta
    memcpy(offset_map_ + str_count_ + i, &str_meta, sizeof(str_meta));
  }
}

std::string_view StringDictionary::getStringFromStorageFast(
    const int string_id) const noexcept {
  const StringIdxEntry* str_meta = offset_map_ + idToIndex(string_id);
  return {payload_map_ + str_meta->off, str_meta->size};
}

StringDictionary::PayloadString StringDictionary::getStringFromStorage(
    const int string_id) const noexcept {
  const StringIdxEntry* str_meta = offset_map_ + idToIndex(string_id);
  if (str_meta->size == 0xffff) {
    // hit the canary
    return {nullptr, 0, true};
  }
  return {payload_map_ + str_meta->off, str_meta->size, false};
}

void StringDictionary::addPayloadCapacity(const size_t min_capacity_requested) noexcept {
  payload_map_ = static_cast<char*>(
      addMemoryCapacity(payload_map_, payload_file_size_, min_capacity_requested));
}

void StringDictionary::addOffsetCapacity(const size_t min_capacity_requested) noexcept {
  offset_map_ = static_cast<StringIdxEntry*>(
      addMemoryCapacity(offset_map_, offset_file_size_, min_capacity_requested));
}

void* StringDictionary::addMemoryCapacity(void* addr,
                                          size_t& mem_size,
                                          const size_t min_capacity_requested) noexcept {
  const size_t canary_buff_size_to_add =
      std::max(static_cast<size_t>(1024 * SYSTEM_PAGE_SIZE),
               (min_capacity_requested / SYSTEM_PAGE_SIZE + 1) * SYSTEM_PAGE_SIZE);
  if (canary_buffer_size < canary_buff_size_to_add) {
    CANARY_BUFFER =
        reinterpret_cast<char*>(realloc(CANARY_BUFFER, canary_buff_size_to_add));
    canary_buffer_size = canary_buff_size_to_add;
    CHECK(CANARY_BUFFER);
    memset(CANARY_BUFFER, 0xff, canary_buff_size_to_add);
  }
  void* new_addr = realloc(addr, mem_size + canary_buff_size_to_add);
  CHECK(new_addr);
  void* write_addr = reinterpret_cast<void*>(static_cast<char*>(new_addr) + mem_size);
  CHECK(memcpy(write_addr, CANARY_BUFFER, canary_buff_size_to_add));
  mem_size += canary_buff_size_to_add;
  return new_addr;
}

void StringDictionary::invalidateInvertedIndex() noexcept {
  if (!like_cache_.empty()) {
    decltype(like_cache_)().swap(like_cache_);
  }
  if (!regex_cache_.empty()) {
    decltype(regex_cache_)().swap(regex_cache_);
  }
  if (!equal_cache_.empty()) {
    decltype(equal_cache_)().swap(equal_cache_);
  }
  compare_cache_.invalidateInvertedIndex();
}

void StringDictionary::buildSortedCache() const {
  // This method is not thread-safe.
  const auto cur_cache_size = sorted_cache.size();
  std::vector<int32_t> temp_sorted_cache;
  for (size_t i = cur_cache_size; i < str_count_; i++) {
    temp_sorted_cache.push_back(indexToId(i));
  }
  sortCache(temp_sorted_cache);
  mergeSortedCache(temp_sorted_cache);
}

void StringDictionary::sortCache(std::vector<int32_t>& cache) const {
  // This method is not thread-safe.

  // this boost sort is creating some problems when we use UTF-8 encoded strings.
  // TODO (vraj): investigate What is wrong with boost sort and try to mitigate it.

  std::sort(cache.begin(), cache.end(), [this](int32_t a, int32_t b) {
    auto a_str = this->getStringFromStorage(a);
    auto b_str = this->getStringFromStorage(b);
    return string_lt(a_str.c_str_ptr, a_str.size, b_str.c_str_ptr, b_str.size);
  });
}

void StringDictionary::mergeSortedCache(std::vector<int32_t>& temp_sorted_cache) const {
  // this method is not thread safe
  std::vector<int32_t> updated_cache(temp_sorted_cache.size() + sorted_cache.size());
  size_t t_idx = 0, s_idx = 0, idx = 0;
  for (; t_idx < temp_sorted_cache.size() && s_idx < sorted_cache.size(); idx++) {
    auto t_string = getStringFromStorage(temp_sorted_cache[t_idx]);
    auto s_string = getStringFromStorage(sorted_cache[s_idx]);
    const auto insert_from_temp_cache =
        string_lt(t_string.c_str_ptr, t_string.size, s_string.c_str_ptr, s_string.size);
    if (insert_from_temp_cache) {
      updated_cache[idx] = temp_sorted_cache[t_idx++];
    } else {
      updated_cache[idx] = sorted_cache[s_idx++];
    }
  }
  while (t_idx < temp_sorted_cache.size()) {
    updated_cache[idx++] = temp_sorted_cache[t_idx++];
  }
  while (s_idx < sorted_cache.size()) {
    updated_cache[idx++] = sorted_cache[s_idx++];
  }
  sorted_cache.swap(updated_cache);
}

std::vector<int32_t> StringDictionary::buildIntersectionTranslationMap(
    const StringDictionary* dest) const {
  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };
  const size_t num_source_strings = entryCount();
  const size_t num_dest_strings = dest->entryCount();
  std::vector<int32_t> translated_ids(num_source_strings);
  StringDictionaryTranslator::buildDictionaryTranslationMap(this,
                                                            dest,
                                                            translated_ids.data(),
                                                            num_source_strings,
                                                            num_dest_strings,
                                                            false,
                                                            dummy_callback);
  return translated_ids;
}

std::vector<int32_t> StringDictionary::buildUnionTranslationMap(
    StringDictionary* dest) const {
  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };
  const size_t num_source_strings = entryCount();
  const size_t num_dest_strings = dest->entryCount();
  std::vector<int32_t> translated_ids(num_source_strings);
  auto num_untranslated_strings =
      StringDictionaryTranslator::buildDictionaryTranslationMap(this,
                                                                dest,
                                                                translated_ids.data(),
                                                                num_source_strings,
                                                                num_dest_strings,
                                                                false,
                                                                dummy_callback);
  if (num_untranslated_strings > 0) {
    // Todo (todd): Add call to fetch string_views (local) or strings (distributed)
    // for all non-translated ids to avoid string-by-string fetch
    for (int32_t source_string_id = 0;
         source_string_id < static_cast<int32_t>(translated_ids.size());
         ++source_string_id) {
      if (translated_ids[source_string_id] == StringDictionary::INVALID_STR_ID) {
        const auto source_string = getStringUnlocked(source_string_id);
        const auto dest_string_id = dest->getOrAdd(source_string);
        translated_ids[source_string_id] = dest_string_id;
      }
    }
  }
  return translated_ids;
}

}  // namespace legacy

std::vector<int32_t> StringDictionaryTranslator::buildDictionaryTranslationMap(
    const std::shared_ptr<StringDictionary> source_dict,
    const std::shared_ptr<StringDictionary> dest_dict,
    StringLookupCallback const& dest_transient_lookup_callback) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t num_source_strings = source_dict->entryCount();
  const size_t num_dest_strings = dest_dict->entryCount();
  std::vector<int32_t> translated_ids(num_source_strings);
  StringDictionaryTranslator::buildDictionaryTranslationMap(
      source_dict.get(),
      dest_dict.get(),
      translated_ids.data(),
      num_source_strings,
      num_dest_strings,
      true,  // Just assume true for dest_has_transients as this
             // function is only used for testing currently
      dest_transient_lookup_callback);
  return translated_ids;
}

size_t StringDictionaryTranslator::buildDictionaryTranslationMap(
    const StringDictionary* source_dict,
    const StringDictionary* dest_dict,
    int32_t* translated_ids,
    const int64_t source_generation,
    const int64_t dest_generation,
    const bool dest_has_transients,
    StringLookupCallback const& dest_transient_lookup_callback) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_GE(source_generation, 0L);
  CHECK_GE(dest_generation, 0L);
  const int64_t num_source_strings = source_generation;
  const int64_t num_dest_strings = dest_generation;

  // We can bail early if there are no source strings to translate
  if (num_source_strings == 0L) {
    return 0;
  }

  if (source_dict == dest_dict) {
    throw std::runtime_error("Cannot translate between a string dictionary and itself.");
  }
  const bool this_dict_is_locked_first = source_dict < dest_dict;
  mapd_shared_lock<mapd_shared_mutex> first_read_lock(
      this_dict_is_locked_first ? source_dict->rw_mutex_ : dest_dict->rw_mutex_);
  mapd_shared_lock<mapd_shared_mutex> second_read_lock(
      this_dict_is_locked_first ? dest_dict->rw_mutex_ : source_dict->rw_mutex_);

  // For both source and destination dictionaries we cap the max
  // entries to be translated/translated to at the supplied
  // generation arguments, if valid (i.e. >= 0), otherwise just the
  // size of each dictionary

  CHECK_LE(num_source_strings, static_cast<int64_t>(source_dict->entryCount()));
  CHECK_LE(num_dest_strings, static_cast<int64_t>(dest_dict->entryCount()));

  // Destination distionary is empty so simply fill the output vector with
  // invalid IDs.
  if (!num_dest_strings) {
    fillInvalidStringIdsParallel(0, num_source_strings, translated_ids);
    return num_source_strings;
  }

  size_t base_num_strings_not_translated = 0;
  if (source_dict->base_dict_) {
    base_num_strings_not_translated = buildDictionaryTranslationMap(
        source_dict->base_dict_.get(),
        dest_dict,
        translated_ids,
        std::min(source_generation, source_dict->base_generation_),
        dest_generation,
        dest_has_transients,
        dest_transient_lookup_callback);
  }

  constexpr int64_t target_strings_per_task{1000};

  // The below logic, by executing low-level private variable accesses on both
  // dictionaries, is less clean than a previous variant that simply called
  // `getStringViews` from the source dictionary and then called `getBulk` on the
  // destination dictionary, but this version gets significantly better performance
  // (~2X), likely due to eliminating the overhead of writing out the string views and
  // then reading them back in (along with the associated cache misses)
  auto total_num_strings_not_translated = tbb::parallel_reduce(
      tbb::blocked_range<int32_t>(source_dict->base_generation_,
                                  num_source_strings,
                                  target_strings_per_task /* tbb grain_size */),
      base_num_strings_not_translated,
      [&](const tbb::blocked_range<int32_t>& r, size_t num_strings_not_translated) {
        const int32_t start_idx = r.begin();
        const int32_t end_idx = r.end();
        for (int32_t source_string_id = start_idx; source_string_id != end_idx;
             ++source_string_id) {
          const std::string_view source_str =
              source_dict->getStringFromStorageFast(source_string_id);
      // Get the hash from this/the source dictionary's cache, as the function
      // will be the same for the dest_dict, sparing us having to recompute it

#ifndef USE_LEGACY_STR_DICT
          const auto translated_string_id = dest_dict->getIdOfString(source_str);
#else
          const auto translated_string_id =
              source_dict->materialize_hashes_
                  ? dest_dict->getUnlocked(source_str,
                                           source_dict->hashById(source_string_id))
                  : dest_dict->getUnlocked(source_str);
#endif
          if (translated_string_id >= 0 && translated_string_id < num_dest_strings) {
            translated_ids[source_string_id] = translated_string_id;
          } else {
            if (!dest_has_transients ||
                dest_transient_lookup_callback(source_str, source_string_id)) {
              translated_ids[source_string_id] = StringDictionary::INVALID_STR_ID;
              num_strings_not_translated++;
            }
          }
        }
        return num_strings_not_translated;
      },
      std::plus<size_t>());
  return total_num_strings_not_translated;
}

namespace fast {

// Functors passed to eachStringSerially() must derive from StringCallback.
// Each std::string const& (if isClient()) or std::string_view (if !isClient())
// plus string_id is passed to the callback functor.
void StringDictionary::eachStringSerially(int64_t const generation,
                                          StringCallback& serial_callback) const {
  // TODO: generation support
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  for (size_t i = 0; i < numStrings(); i++) {
    serial_callback(str(i), i);
  }
}

int32_t StringDictionary::getOrAdd(const std::string_view& str) noexcept {
  if (str.size() == 0) {
    return inline_int_null_value<int32_t>();
  }
  CHECK(str.size() <= MAX_STRLEN);
  const uint32_t hash = hash_string(str);

  mapd_unique_lock<mapd_shared_mutex> rw_lock(rw_mutex_);
  const int32_t string_id = addString(hash, str);
  return string_id;
}

// can't we just do a default argument here?
template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec) const {
  return getBulk(string_vec, encoded_vec, -1L /* generation */);
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec) const;

template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec,
                                 const int64_t generation) const {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, string_vec.size()),
      size_t(0),
      [&](const tbb::blocked_range<size_t>& r, const auto init) {
        size_t num_strings_not_found = init;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          const auto& str = string_vec[i];
          if (str.empty()) {
            encoded_vec[i] = inline_int_null_value<T>();
          } else {
            if (str.size() > StringDictionary::MAX_STRLEN) {
              legacy::throw_string_too_long_error(str, dict_ref_);
            }

            const auto hash = hash_string(str);
            const auto string_id = getIdOfStringImpl(hash, str);
            if (string_id == StringDictionary::INVALID_STR_ID ||
                string_id > int32_t(strings.size())) {
              encoded_vec[i] = StringDictionary::INVALID_STR_ID;
              num_strings_not_found++;
            }
            encoded_vec[i] = string_id;
          }
        }
        return num_strings_not_found;
      },
      std::plus<size_t>());
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec,
                                          const int64_t generation) const;

template <class T, class String>
void StringDictionary::getOrAddBulk(const std::vector<String>& string_vec,
                                    T* output_string_ids) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);

  // compute hashes
  auto hashes = std::make_unique<uint32_t[]>(string_vec.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(
                        0,
                        string_vec.size(),
                        /*grain_size=*/25'000),  // experimentally determined on taxi
                                                 // benchmark, may need tweaking
                    [&string_vec, &hashes](const tbb::blocked_range<size_t>& r) {
                      for (size_t curr_id = r.begin(); curr_id != r.end(); ++curr_id) {
                        if (string_vec[curr_id].empty()) {
                          continue;
                        }
                        hashes[curr_id] = hash_string(string_vec[curr_id]);
                      }
                    });

  for (size_t i = 0; i < string_vec.size(); i++) {
    const auto& input_string = string_vec[i];
    if (input_string.empty()) {
      output_string_ids[i] = inline_int_null_value<T>();
    } else {
      // add string to storage and store id
      const auto& hash = hashes[i];
      output_string_ids[i] = addString(hash, input_string);
    }
  }
}

template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             int32_t* encoded_vec);

template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    int32_t* encoded_vec);

template <class String>
int32_t StringDictionary::getIdOfString(const String& str) const {
  if (str.size() == 0) {
    return inline_int_null_value<int32_t>();
  }
  CHECK(str.size() <= MAX_STRLEN);
  const uint32_t hash = hash_string(str);

  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getIdOfStringImpl(hash, str);
}

template int32_t StringDictionary::getIdOfString(const std::string&) const;
template int32_t StringDictionary::getIdOfString(const std::string_view&) const;

std::string StringDictionary::getString(int32_t string_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK_LT(string_id, static_cast<int32_t>(numStrings()));
  return str(string_id);
}

std::pair<char*, size_t> StringDictionary::getStringBytes(
    int32_t string_id) const noexcept {
  CHECK(false);
  return std::make_pair(nullptr, 0);
}

size_t StringDictionary::storageEntryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return numStrings();
}

std::vector<int32_t> StringDictionary::getLike(const std::string& pattern,
                                               const bool icase,
                                               const bool is_simple,
                                               const char escape,
                                               const size_t generation) const {
  CHECK(false);
  return {};
}

std::vector<int32_t> StringDictionary::getCompare(const std::string& pattern,
                                                  const std::string& comp_operator,
                                                  const size_t generation) {
  CHECK(false);
  return {};
}

std::vector<int32_t> StringDictionary::getRegexpLike(const std::string& pattern,
                                                     const char escape,
                                                     const size_t generation) const {
  CHECK(false);
  return {};
}

std::vector<std::string> StringDictionary::copyStrings() const {
  CHECK(false);
  return {};
}

int32_t StringDictionary::getUnlocked(const std::string_view sv) const noexcept {
  const uint32_t hash = hash_string(sv);
  return getIdOfStringImpl(hash, sv);
}

std::string_view StringDictionary::getStringFromStorageFast(
    const int string_id) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return str(string_id);
}

template <class String>
int32_t StringDictionary::addString(const uint32_t hash, const String& input_string) {
  const size_t hash_table_size = hash_to_id_map.size();
  uint32_t bucket = hash & (hash_table_size - 1);
  // find an empty slot in the hash map
  while (true) {
    const int32_t candidate_string_id = hash_to_id_map[bucket].string_id;
    if (candidate_string_id == INVALID_STR_ID) {
      // found an open slot
      // found an open slot - add the string to the strings payload
      const auto str_id = addStringToMaps(bucket, hash, input_string);
      if (2 * str_id > int32_t(size())) {
        resize(2 * size());
        VLOG(3) << "Resized to " << size() << " (holds " << numStrings() << ")";
      }

      return str_id;
    }

    // slot is full, check for a collision
    if (hash == hash_to_id_map[bucket].hash) {
      const auto& existing_string = hash_to_id_map[bucket].string;
      if (existing_string == input_string) {
        // found an existing string that matches
        break;
      }
    }

    // wrap around
    if (++bucket == hash_table_size) {
      bucket = 0;
    }
  }
  return hash_to_id_map[bucket].string_id;
}

// on resize we need to re-hash the strings, as the hash is based on the total hash table
// size
// NOTE: in taxi this never gets called b/c the dictionaries are so small
void StringDictionary::resize(const size_t new_size) {
  CHECK_GT(new_size, size());

  strings.reserve(new_size);

  std::vector<HashMapPayload> new_hash_map(new_size);
  hash_to_id_map.swap(new_hash_map);
  const size_t hash_table_size = hash_to_id_map.size();

  for (size_t i = 0; i < numStrings(); i++) {
    const auto& crt_str = *strings[i].get();
    const auto hash = hash_string(crt_str);
    uint32_t bucket = hash & (hash_table_size - 1);

    while (true) {
      const int32_t candidate_string_id = hash_to_id_map[bucket].string_id;
      if (candidate_string_id == INVALID_STR_ID) {
        // found an open slot
        hash_to_id_map[bucket].set(i, hash, crt_str);
        break;
      }

      // slot is full, check for a collision
      if (hash == hash_to_id_map[bucket].hash) {
        const auto& existing_string = hash_to_id_map[bucket].string;
        if (existing_string == crt_str) {
          // found an existing string that matches
          LOG(WARNING)
              << "Found an existing string that matches during str dict hash table "
                 "resize. Str dict may contain duplicates. Existing string: "
              << existing_string << ", Input string: " << crt_str;
          break;
        }
      }

      // wrap around
      if (++bucket == hash_table_size) {
        bucket = 0;
      }
    }
  }
}

int32_t StringDictionary::getIdOfStringImpl(const uint32_t hash,
                                            const std::string_view input_string) const {
  const size_t hash_table_size = hash_to_id_map.size();
  uint32_t bucket = hash & (hash_table_size - 1);
  // find an empty slot in the hash map
  while (true) {
    const int32_t candidate_string_id = hash_to_id_map[bucket].string_id;
    if (candidate_string_id == INVALID_STR_ID) {
      // found an open slot
      return candidate_string_id;
    }

    // slot is full, check for a collision
    if (hash == hash_to_id_map[bucket].hash) {
      const auto& existing_string = hash_to_id_map[bucket].string;
      if (existing_string == input_string) {
        // found an existing string that matches
        return candidate_string_id;
      }
    }

    // wrap around
    if (++bucket == hash_table_size) {
      bucket = 0;
    }
  }
  CHECK(false);
  return -1;
}

}  // namespace fast
