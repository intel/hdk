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

#ifndef STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
#define STRINGDICTIONARY_STRINGDICTIONARYPROXY_H

#include "Logger/Logger.h"  // For CHECK macros
#include "Shared/funcannotations.h"
#include "StringDictionary.h"

#include <map>
#include <optional>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifndef STRINGDICTIONARY_EXPORT
#define STRINGDICTIONARY_EXPORT EXTERN
#endif

// used to access a StringDictionary when transient strings are involved
class StringDictionaryProxy {
 public:
  StringDictionaryProxy(StringDictionaryProxy const&) = delete;
  StringDictionaryProxy const& operator=(StringDictionaryProxy const&) = delete;
  StringDictionaryProxy(std::shared_ptr<StringDictionary> sd,
                        const int64_t generation = -1);

  bool operator==(StringDictionaryProxy const&) const;
  bool operator!=(StringDictionaryProxy const&) const;

  // enum SetOp { kUnion = 0, kIntersection };

  StringDictionary* getBaseDictionary() const noexcept;
  int64_t getBaseGeneration() const noexcept;

  /**
   * @brief Executes read-only lookup of a vector of strings and returns a vector of their
   integer ids
  *
  * This function, unlike getOrAddTransientBulk, will not add strings to the dictionary.
  * Use this function if strings that don't currently exist in the StringDictionaryProxy
  * should not be added to the proxy as transient entries.
  * This method also has performance advantages over getOrAddTransientBulk for read-only
  * use cases, in that it can:
  * 1) Take a read lock instead of a write lock for the transient lookups
  * 2) Use a tbb::parallel_for implementation of the transient string lookups as
  * we are guaranteed that the underlying map of strings to int ids cannot change

  * @param strings - Vector of strings to perform string id lookups on
  * @return A vector of string_ids of the same length as strings, containing
  * the id of any strings for which were found in the underlying StringDictionary
  * instance or in the proxy's tranient map, otherwise
  * StringDictionary::INVALID_STR_ID for strings not found.
  */

  std::vector<int32_t> getBulk(const std::vector<std::string>& strings) const;
  int32_t getOrAdd(const std::string& str);
  // Not currently used
  std::vector<int32_t> getOrAddBulk(const std::vector<std::string>& strings);
  int32_t getIdOfString(const std::string& str) const;
  std::string getString(int32_t string_id) const;
  std::vector<std::string> getStrings(const std::vector<int32_t>& string_ids) const;
  std::pair<const char*, size_t> getStringBytes(int32_t string_id) const noexcept;

  /**
   * @brief Builds a vectorized string_id translation map from this proxy to dest_proxy
   *
   * @param dest_proxy StringDictionaryProxy that we are to map this proxy's string ids to
   *
   * @return A std::vector<int32_t> of string ids for both transient and non-transient
   * strings, mapping to their translated string_ids.
   */
  std::vector<int32_t> buildIntersectionTranslationMap(
      const StringDictionaryProxy* dest_proxy) const;

  std::vector<int32_t> buildUnionTranslationMapToOtherProxy(
      StringDictionaryProxy* dest_proxy) const;

  /**
   * @brief Returns the number of transient string entries for this proxy,
   *
   * @return size_t Number of transient string entries for this proxy
   *
   */
  size_t transientEntryCount() const;

  /**
   * @brief Returns the number of total string entries for this proxy, both stored
   * in the underlying dictionary and in the transient map. Equal to
   * storageEntryCount() + transientEntryCount()
   *
   * @return size_t Number of total string entries for this proxy
   *
   */
  STRINGDICTIONARY_EXPORT size_t entryCount() const;

  void updateGeneration(const int64_t generation) noexcept;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape) const;

  std::vector<int32_t> getCompare(const std::string& pattern,
                                  const std::string& comp_operator) const;

  std::vector<int32_t> getRegexpLike(const std::string& pattern, const char escape) const;

  // The std::string must live in the map, and std::string const* in the vector. As
  // desirable as it might be to have it the other way, string addresses won't change
  // in the std::map when new strings are added, but will change in a std::vector.
  using TransientMap = std::map<std::string, int32_t, std::less<>>;

  std::vector<std::string> copyStrings() const;

  // Iterate over transient strings, then non-transients.
  void eachStringSerially(StringDictionary::StringCallback&) const;

 private:
  unsigned transientIdToIndex(int32_t const id) const {
    return static_cast<unsigned>(id - generation_);
  }

  int32_t transientIndexToId(unsigned const index) const { return generation_ + index; }

  /**
   * @brief Returns the number of string entries in the underlying string dictionary,
   * at this proxy's generation_ if it is set/valid, otherwise just the current
   * size of the dictionary
   *
   * @return size_t Number of entries in the string dictionary
   * (at this proxy's generation if set)
   *
   */
  size_t storageEntryCount() const;

  std::string getStringUnlocked(const int32_t string_id) const;
  size_t transientEntryCountUnlocked() const;
  size_t entryCountUnlocked() const;
  template <typename String>
  int32_t lookupTransientStringUnlocked(const String& lookup_string) const;
  size_t getTransientBulkImpl(const std::vector<std::string>& strings,
                              int32_t* string_ids,
                              const bool take_read_lock) const;
  template <typename String>
  size_t transientLookupBulk(const std::vector<String>& lookup_strings,
                             int32_t* string_ids,
                             const bool take_read_lock) const;
  template <typename String>
  size_t transientLookupBulkUnlocked(const std::vector<String>& lookup_strings,
                                     int32_t* string_ids) const;
  template <typename String>
  size_t transientLookupBulkParallelUnlocked(const std::vector<String>& lookup_strings,
                                             int32_t* string_ids) const;

  std::vector<int32_t> buildIntersectionTranslationMapToOtherProxyUnlocked(
      const StringDictionaryProxy* dest_proxy,
      size_t& num_strings_not_translated) const;
  std::shared_ptr<StringDictionary> string_dict_;
  TransientMap transient_str_to_int_;
  // Holds pointers into transient_str_to_int_
  std::vector<std::string const*> transient_string_vec_;
  int64_t generation_;
  mutable std::shared_mutex rw_mutex_;

  // Return INVALID_STR_ID if not found on string_dict_. Don't lock or check transients.
  template <typename String>
  int32_t getIdOfStringFromClient(String const&) const;
  template <typename String>
  int32_t getOrAddTransientUnlocked(String const&);

  friend class StringLocalCallback;
};

class StringDictionaryProxyProvider {
 public:
  virtual StringDictionaryProxy* getStringDictionaryProxy(
      const int dict_id,
      const bool with_generation) const = 0;
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
