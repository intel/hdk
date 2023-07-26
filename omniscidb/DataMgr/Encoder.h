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

#ifndef ENCODER_H
#define ENCODER_H

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ChunkMetadata.h"
#include "IR/Type.h"
#include "Shared/DateConverters.h"
#include "Shared/sqltypes.h"
#include "Shared/types.h"

namespace Data_Namespace {
class AbstractBuffer;
}

// default max input buffer size to 1MB
#define MAX_INPUT_BUF_SIZE 1048576

class DecimalOverflowValidator {
 public:
  DecimalOverflowValidator(const hdk::ir::Type* type) {
    if (type && type->isArray()) {
      type = type->as<hdk::ir::ArrayBaseType>()->elemType();
    }

    if (type && type->isDecimal()) {
      do_check_ = true;
      int precision = type->as<hdk::ir::DecimalType>()->precision();
      int scale = type->as<hdk::ir::DecimalType>()->scale();
      max_ = (int64_t)std::pow((double)10.0, precision);
      min_ = -max_;
      pow10_ = precision - scale;

    } else {
      do_check_ = false;
      max_ = 1;
      min_ = -1;
      pow10_ = 0;
    }
  }

  template <typename T>
  bool validate(T value) const {
    if (std::is_integral<T>::value) {
      return do_validate(static_cast<int64_t>(value));
    }
    // return true if data is not supported by this validator
    return true;
  }

  // returns is valid
  bool do_validate(int64_t value) const {
    if (!do_check_) {
      return true;
    }

    if (value >= max_ || value <= min_) {
      return false;
    }
    return true;
  }

 private:
  bool do_check_;
  int64_t max_;
  int64_t min_;
  int pow10_;
};

template <typename INNER_VALIDATOR>
class NullAwareValidator {
 public:
  NullAwareValidator(const hdk::ir::Type* type, INNER_VALIDATOR* inner_validator) {
    if (type && type->isArray()) {
      type = type->as<hdk::ir::ArrayBaseType>()->elemType();
    }

    skip_null_check_ = !type || !type->nullable();
    inner_validator_ = inner_validator;
  }

  template <typename T>
  void validate(T value) {
    if (skip_null_check_ || value != inline_int_null_value<T>()) {
      inner_validator_->template validate<T>(value);
    }
  }

 private:
  bool skip_null_check_;
  INNER_VALIDATOR* inner_validator_;
};

class DateDaysOverflowValidator {
 public:
  DateDaysOverflowValidator(const hdk::ir::Type* type) {
    if (type && type->isArray()) {
      type = type->as<hdk::ir::ArrayBaseType>()->elemType();
    }

    bool is_date_16_;
    if (type && type->isDate()) {
      is_date_in_days_ = type->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay;
      is_date_16_ = type->size() == 2;
    } else {
      is_date_in_days_ = false;
      is_date_16_ = false;
    }
    max_ = is_date_16_ ? static_cast<int64_t>(std::numeric_limits<int16_t>::max())
                       : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    min_ = is_date_16_ ? static_cast<int64_t>(std::numeric_limits<int16_t>::min())
                       : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  }

  template <typename T>
  void validate(T value) {
    if (!is_date_in_days_ || !std::is_integral<T>::value) {
      return;
    }
    const int64_t days =
        DateConverters::get_epoch_days_from_seconds(static_cast<int64_t>(value));
    if (days > max_) {
      throw std::runtime_error("Date encoding overflow: Epoch days " +
                               std::to_string(days) + " greater than maximum capacity " +
                               std::to_string(max_));
    }
    if (days < min_) {
      throw std::runtime_error("Date encoding underflow: Epoch days " +
                               std::to_string(days) + " less than minimum capacity " +
                               std::to_string(min_));
    }
  }

 private:
  bool is_date_in_days_;
  int64_t max_;
  int64_t min_;
};

class Encoder {
 public:
  static Encoder* Create(Data_Namespace::AbstractBuffer* buffer,
                         const hdk::ir::Type* type);
  Encoder(Data_Namespace::AbstractBuffer* buffer);
  virtual ~Encoder() {}

  virtual std::shared_ptr<ChunkMetadata> getMetadata();
  // Only called from the executor for synthesized meta-information.
  virtual std::shared_ptr<ChunkMetadata> getMetadata(const hdk::ir::Type* type) = 0;
  virtual void updateStats(const int64_t val, const bool is_null) = 0;
  virtual void updateStats(const double val, const bool is_null) = 0;

  /**
   * Update statistics for data without appending.
   *
   * @param src_data - the data with which to update statistics
   * @param num_elements - the number of elements to scan in the data
   */
  virtual void updateStats(const int8_t* const src_data, const size_t num_elements) = 0;

  /**
   * Update statistics for encoded data without appending.
   *
   * @param dst_data - the data with which to update statistics
   * @param num_elements - the number of elements to scan in the data
   * @param fixlen_array - true if stats are computed for fixlen array
   */
  virtual void updateStatsEncoded(const int8_t* const dst_data,
                                  const size_t num_elements,
                                  bool fixlen_array = false) {
    UNREACHABLE();
  }

  /**
   * Update statistics for string data without appending.
   *
   * @param src_data - the string data with which to update statistics
   * @param start_idx - the offset into `src_data` to start the update
   * @param num_elements - the number of elements to scan in the string data
   */
  virtual void updateStats(const std::vector<std::string>* const src_data,
                           const size_t start_idx,
                           const size_t num_elements) = 0;

  /**
   * Update statistics for array data without appending.
   *
   * @param src_data - the array data with which to update statistics
   * @param start_idx - the offset into `src_data` to start the update
   * @param num_elements - the number of elements to scan in the array data
   */
  virtual void updateStats(const std::vector<ArrayDatum>* const src_data,
                           const size_t start_idx,
                           const size_t num_elements) = 0;

  virtual void reduceStats(const Encoder&) = 0;
  virtual void copyMetadata(const Encoder* copyFromEncoder) = 0;
  virtual void writeMetadata(FILE* f /*, const size_t offset*/) = 0;
  virtual void readMetadata(FILE* f /*, const size_t offset*/) = 0;

  /**
   * @brief: Reset chunk level stats (min, max, nulls) using new values from the argument.
   * @return: True if an update occurred and the chunk needs to be flushed. False
   * otherwise. Default false if metadata update is unsupported. Only reset chunk stats if
   * the incoming stats differ from the current stats.
   */
  virtual bool resetChunkStats(const ChunkStats&) {
    UNREACHABLE() << "Attempting to reset stats for unsupported type.";
    return false;
  }

  /**
   * Resets chunk metadata stats to their default values.
   */
  virtual void resetChunkStats() = 0;

  /**
   * @brief: Copy current chunk level stats (min, max, nulls) to the output arg.
   */
  virtual void fillChunkStats(ChunkStats&, const hdk::ir::Type*) {
    UNREACHABLE() << "Attempting to get stats for unsupported type.";
  }

  size_t getNumElems() const { return num_elems_; }
  void setNumElems(const size_t num_elems) { num_elems_ = num_elems; }

 protected:
  size_t num_elems_;

  Data_Namespace::AbstractBuffer* buffer_;

  DecimalOverflowValidator decimal_overflow_validator_;
  DateDaysOverflowValidator date_days_overflow_validator_;
};

#endif  // Encoder_h
