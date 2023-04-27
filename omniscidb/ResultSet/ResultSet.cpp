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

/**
 * @file    ResultSet.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 */

#include "CountDistinct.h"
#include "RowSetMemoryOwner.h"

#include "DataMgr/Allocators/GpuAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "ResultSet/ResultSet.h"
#include "Shared/InlineNullValues.h"
#include "Shared/Intervals.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"
#include "Shared/threading.h"
#include "Utils/ExtractFromTime.h"

#include <boost/math/special_functions/fpclassify.hpp>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <future>
#include <numeric>

constexpr int64_t uninitialized_cached_row_count{-1};

void ResultSet::keepFirstN(const size_t n) {
  invalidateCachedRowCount();
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  invalidateCachedRowCount();
  drop_first_ = n;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     Data_Namespace::DataMgr* data_mgr,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(-1)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , data_mgr_(data_mgr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count) {}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                     const std::vector<std::vector<int64_t>>& frag_offsets,
                     const std::vector<int64_t>& consistent_frag_sizes,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     Data_Namespace::DataMgr* data_mgr,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , lazy_fetch_info_(lazy_fetch_info)
    , col_buffers_{col_buffers}
    , frag_offsets_{frag_offsets}
    , consistent_frag_sizes_{consistent_frag_sizes}
    , data_mgr_(data_mgr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count) {}

ResultSet::ResultSet(const std::shared_ptr<const hdk::ir::Estimator> estimator,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     Data_Namespace::DataMgr* data_mgr)
    : device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_{}
    , crt_row_buff_idx_(0)
    , estimator_(estimator)
    , data_mgr_(data_mgr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count) {
  if (device_type == ExecutorDeviceType::GPU) {
    device_estimator_buffer_ = GpuAllocator::allocGpuAbstractBuffer(
        getBufferProvider(), estimator_->getBufferSize(), device_id_);
    getBufferProvider()->zeroDeviceMem(device_estimator_buffer_->getMemoryPtr(),
                                       estimator_->getBufferSize(),
                                       device_id_);
  } else {
    host_estimator_buffer_ =
        static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  }
}

ResultSet::ResultSet(const std::string& explanation)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , separate_varlen_storage_valid_(false)
    , explanation_(explanation)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count) {}

ResultSet::ResultSet(int64_t queue_time_ms,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , timings_(QueryExecutionTimings{queue_time_ms, 0, 0})
    , separate_varlen_storage_valid_(false)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(uninitialized_cached_row_count) {}

ResultSet::~ResultSet() {
  if (storage_) {
    if (!storage_->buff_is_provided_) {
      CHECK(storage_->getUnderlyingBuffer());
      free(storage_->getUnderlyingBuffer());
    }
  }
  for (auto& storage : appended_storage_) {
    if (storage && !storage->buff_is_provided_) {
      free(storage->getUnderlyingBuffer());
    }
  }
  if (host_estimator_buffer_) {
    CHECK(device_type_ == ExecutorDeviceType::CPU || device_estimator_buffer_);
    free(host_estimator_buffer_);
  }
  if (device_estimator_buffer_) {
    CHECK(data_mgr_);
    data_mgr_->free(device_estimator_buffer_);
  }
}

std::string ResultSet::getStrScalarVal(const ScalarTargetValue& current_scalar,
                                       const hdk::ir::Type* col_type) const {
  std::ostringstream oss;
  if (current_scalar.type() == typeid(NullableString)) {
    if (boost::get<NullableString>(current_scalar).type() == typeid(void*)) {
      oss << "null";
    } else {
      oss << boost::get<std::string>(boost::get<NullableString>(current_scalar));
    }
  } else {
    if (col_type->isExtDictionary()) {
      const int32_t dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();
      const auto sdp = getStringDictionaryProxy(dict_id);
      oss << "idx:" << ((sdp->storageEntryCount()) ? current_scalar : "null") << ", str:"
          << "\"" << sdp->getString(boost::get<int64_t>(current_scalar)) << "\"";
    } else {
      if ((col_type->isInt64() &&
           boost::get<int64_t>(current_scalar) == std::numeric_limits<int64_t>::min()) ||
          (col_type->isInt32() &&
           boost::get<int64_t>(current_scalar) == std::numeric_limits<int32_t>::min()) ||
          (col_type->isInt16() &&
           boost::get<int64_t>(current_scalar) == std::numeric_limits<int16_t>::min()) ||
          (col_type->isInt8() &&
           boost::get<int64_t>(current_scalar) == std::numeric_limits<int8_t>::min()) ||
          (col_type->isFp32() &&
           boost::get<float>(current_scalar) == std::numeric_limits<float>::min()) ||
          (col_type->isFp64() &&
           boost::get<double>(current_scalar) == std::numeric_limits<double>::min()) ||
          (col_type->isDateTime() &&
           boost::get<int64_t>(current_scalar) == std::numeric_limits<int64_t>::min())) {
        oss << "null";
      } else {
        if (col_type->isDate()) {
          oss << getStrDateFromSeconds(boost::get<int64_t>(current_scalar));
        } else if (col_type->isTime()) {
          oss << getStrTimeFromSeconds(boost::get<int64_t>(current_scalar));
        } else if (col_type->isTimestamp()) {
          oss << getStrTStamp(boost::get<int64_t>(current_scalar),
                              col_type->as<hdk::ir::DateTimeBaseType>()->unit());
        } else {
          oss << current_scalar;
        }
      }
    }
  }
  return oss.str();
}

std::string ResultSet::contentToString(bool header) const {
  std::ostringstream oss;
  constexpr char col_delimiter = '|';
  if (header) {
    oss << "Column types:\n";
    for (size_t col = 0; col < colCount(); col++) {
      if (col) {
        oss << col_delimiter;
      }
      oss << colType(col)->toString();
    }
    oss << "\nData:\n";
  }
  moveToBegin();
  while (true) {
    const auto row = getNextRow(false, false);
    if (row.empty()) {
      break;
    } else {
      for (size_t col_idx = 0; col_idx < row.size(); col_idx++) {
        if (col_idx) {
          oss << col_delimiter;
        }
        if (row[col_idx].type() == typeid(ScalarTargetValue)) {
          const auto scalar_col_val = boost::get<ScalarTargetValue>(row[col_idx]);
          oss << getStrScalarVal(scalar_col_val, colType(col_idx));
        } else {
          const auto array_col_val = boost::get<ArrayTargetValue>(&row[col_idx]);
          if (!array_col_val->is_initialized()) {
            oss << "null";
          } else {
            const auto& scalar_vector = array_col_val->get();
            oss << "[";
            for (const auto& scalar_val : scalar_vector) {
              oss << getStrScalarVal(scalar_val, colType(col_idx)) << ",";
            }
            oss << "]";
          }
        }
      }
      oss << "\n";
    }
  }
  return oss.str();
}

std::string ResultSet::summaryToString() const {
  std::ostringstream oss;
  oss << "Result Set Info" << std::endl;
  oss << "\tLayout: " << query_mem_desc_.queryDescTypeToString() << std::endl;
  oss << "\tColumns: " << colCount() << std::endl;
  oss << "\tRows: " << rowCount() << std::endl;
  oss << "\tEntry count: " << entryCount() << std::endl;
  const std::string is_empty = isEmpty() ? "True" : "False";
  oss << "\tIs empty: " << is_empty << std::endl;
  const std::string did_output_columnar = didOutputColumnar() ? "True" : "False;";
  oss << "\tColumnar: " << did_output_columnar << std::endl;
  oss << "\tLazy-fetched columns: " << getNumColumnsLazyFetched() << std::endl;
  const std::string is_direct_columnar_conversion_possible =
      isDirectColumnarConversionPossible() ? "True" : "False";
  oss << "\tDirect columnar conversion possible: "
      << is_direct_columnar_conversion_possible << std::endl;

  size_t num_columns_zero_copy_columnarizable{0};
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    if (isZeroCopyColumnarConversionPossible(target_idx)) {
      num_columns_zero_copy_columnarizable++;
    }
  }
  oss << "\tZero-copy columnar conversion columns: "
      << num_columns_zero_copy_columnarizable << std::endl;

  oss << "\tPermutation size: " << permutation_.size() << std::endl;
  oss << "\tLimit: " << keep_first_ << std::endl;
  oss << "\tOffset: " << drop_first_ << std::endl;
  return oss.str();
}

ExecutorDeviceType ResultSet::getDeviceType() const {
  return device_type_;
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  auto buff = row_set_mem_owner_->allocate(
      query_mem_desc_.getBufferSizeBytes(device_type_), /*thread_idx=*/0);
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    int8_t* buff,
    const std::vector<int64_t>& target_init_vals,
    std::shared_ptr<VarlenOutputInfo> varlen_output_info) const {
  CHECK(buff);
  CHECK(!storage_);
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, true));
  // TODO: add both to the constructor
  storage_->target_init_vals_ = target_init_vals;
  if (varlen_output_info) {
    storage_->varlen_output_info_ = varlen_output_info;
  }
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  auto buff = row_set_mem_owner_->allocate(
      query_mem_desc_.getBufferSizeBytes(device_type_), /*thread_idx=*/0);
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

size_t ResultSet::getCurrentRowBufferIndex() const {
  if (crt_row_buff_idx_ == 0) {
    throw std::runtime_error("current row buffer iteration index is undefined");
  }
  return crt_row_buff_idx_ - 1;
}

// Note: that.appended_storage_ does not get appended to this.
void ResultSet::append(ResultSet& that) {
  invalidateCachedRowCount();
  if (!that.storage_) {
    return;
  }
  appended_storage_.push_back(std::move(that.storage_));
  query_mem_desc_.setEntryCount(
      query_mem_desc_.getEntryCount() +
      appended_storage_.back()->query_mem_desc_.getEntryCount());
  chunks_.insert(chunks_.end(), that.chunks_.begin(), that.chunks_.end());
  col_buffers_.insert(
      col_buffers_.end(), that.col_buffers_.begin(), that.col_buffers_.end());
  frag_offsets_.insert(
      frag_offsets_.end(), that.frag_offsets_.begin(), that.frag_offsets_.end());
  consistent_frag_sizes_.insert(consistent_frag_sizes_.end(),
                                that.consistent_frag_sizes_.begin(),
                                that.consistent_frag_sizes_.end());
  chunk_iters_.insert(
      chunk_iters_.end(), that.chunk_iters_.begin(), that.chunk_iters_.end());
  if (separate_varlen_storage_valid_) {
    CHECK(that.separate_varlen_storage_valid_);
    serialized_varlen_buffer_.insert(serialized_varlen_buffer_.end(),
                                     that.serialized_varlen_buffer_.begin(),
                                     that.serialized_varlen_buffer_.end());
  }
  for (auto& buff : that.literal_buffers_) {
    literal_buffers_.push_back(std::move(buff));
  }
}

const ResultSetStorage* ResultSet::getStorage(size_t idx) const {
  if (!idx) {
    return storage_.get();
  }
  CHECK_LE(idx, appended_storage_.size());
  return appended_storage_[idx - 1].get();
}

size_t ResultSet::getStorageCount() const {
  return storage_.get() ? 1 + appended_storage_.size() : 0;
}

size_t ResultSet::colCount() const {
  return just_explain_ ? 1 : targets_.size();
}

const hdk::ir::Type* ResultSet::colType(const size_t col_idx) const {
  if (just_explain_) {
    return hdk::ir::Context::defaultCtx().text();
  }
  CHECK_LT(col_idx, targets_.size());
  return targets_[col_idx].agg_kind == hdk::ir::AggType::kAvg
             ? hdk::ir::Context::defaultCtx().fp64()
             : targets_[col_idx].type;
}

void ResultSet::setColNames(std::vector<std::string> fields) {
  if (just_explain_) {
    CHECK_EQ((size_t)1, fields.size());
  } else {
    CHECK_EQ(targets_.size(), fields.size());
  }
  fields_ = std::move(fields);
}

bool ResultSet::hasColNames() const {
  return targets_.size() == fields_.size();
}

std::string ResultSet::colName(size_t col_idx) const {
  if (fields_.empty()) {
    return "col" + std::to_string(col_idx);
  } else {
    CHECK_LT(col_idx, fields_.size());
    return fields_[col_idx];
  }
}

StringDictionaryProxy* ResultSet::getStringDictionaryProxy(int const dict_id) const {
  return row_set_mem_owner_->getOrAddStringDictProxy(dict_id);
}

class ResultSet::CellCallback {
  StringDictionaryProxy::IdMap const id_map_;
  int64_t const null_int_;

 public:
  CellCallback(StringDictionaryProxy::IdMap&& id_map, int64_t const null_int)
      : id_map_(std::move(id_map)), null_int_(null_int) {}
  void operator()(int8_t const* const cell_ptr) const {
    using StringId = int32_t;
    StringId* const string_id_ptr =
        const_cast<StringId*>(reinterpret_cast<StringId const*>(cell_ptr));
    if (*string_id_ptr != null_int_) {
      *string_id_ptr = id_map_[*string_id_ptr];
    }
  }
};

void ResultSet::translateDictEncodedColumns(std::vector<TargetInfo> const& targets,
                                            size_t const start_idx) {
  if (storage_) {
    CHECK_EQ(targets.size(), storage_->targets_.size());
    RowIterationState state;
    for (size_t target_idx = start_idx; target_idx < targets.size(); ++target_idx) {
      auto type_lhs = targets[target_idx].type;
      if (type_lhs->isExtDictionary()) {
        auto type_rhs = storage_->targets_[target_idx].type;
        CHECK(type_rhs->isExtDictionary());
        auto lhs_dict_id = type_lhs->as<hdk::ir::ExtDictionaryType>()->dictId();
        auto rhs_dict_id = type_rhs->as<hdk::ir::ExtDictionaryType>()->dictId();
        if (lhs_dict_id != rhs_dict_id) {
          auto* const sdp_lhs = getStringDictionaryProxy(lhs_dict_id);
          CHECK(sdp_lhs);
          auto const* const sdp_rhs = getStringDictionaryProxy(rhs_dict_id);
          CHECK(sdp_rhs);
          state.cur_target_idx_ = target_idx;
          CellCallback const translate_string_ids(sdp_lhs->transientUnion(*sdp_rhs),
                                                  inline_int_null_value(type_rhs));
          eachCellInColumn(state, translate_string_ids);
          const_cast<TargetInfo&>(storage_->targets_[target_idx]).type =
              type_rhs->ctx().extDict(
                  type_rhs->as<hdk::ir::ExtDictionaryType>()->elemType(),
                  lhs_dict_id,
                  type_rhs->size());
        }
      }
    }
  }
}

// For each cell in column target_idx, callback func with pointer to datum.
// This currently assumes the column type is a dictionary-encoded string, but this logic
// can be generalized to other types.
void ResultSet::eachCellInColumn(RowIterationState& state, CellCallback const& func) {
  size_t const target_idx = state.cur_target_idx_;
  QueryMemoryDescriptor& storage_qmd = storage_->query_mem_desc_;
  CHECK_LT(target_idx, lazy_fetch_info_.size());
  auto& col_lazy_fetch = lazy_fetch_info_[target_idx];
  CHECK(col_lazy_fetch.is_lazily_fetched);
  int const target_size = storage_->targets_[target_idx].type->size();
  CHECK_LT(0, target_size) << storage_->targets_[target_idx].toString();
  size_t const nrows = storage_->binSearchRowCount();
  if (storage_qmd.didOutputColumnar()) {
    // Logic based on ResultSet::ColumnWiseTargetAccessor::initializeOffsetsForStorage()
    if (state.buf_ptr_ == nullptr) {
      state.buf_ptr_ = get_cols_ptr(storage_->buff_, storage_qmd);
      state.compact_sz1_ = storage_qmd.getPaddedSlotWidthBytes(state.agg_idx_)
                               ? storage_qmd.getPaddedSlotWidthBytes(state.agg_idx_)
                               : query_mem_desc_.getEffectiveKeyWidth();
    }
    for (size_t j = state.prev_target_idx_; j < state.cur_target_idx_; ++j) {
      size_t const next_target_idx = j + 1;  // Set state to reflect next target_idx j+1
      state.buf_ptr_ = advance_to_next_columnar_target_buff(
          state.buf_ptr_, storage_qmd, state.agg_idx_);
      auto const& next_agg_info = storage_->targets_[next_target_idx];
      state.agg_idx_ =
          advance_slot(state.agg_idx_, next_agg_info, separate_varlen_storage_valid_);
      state.compact_sz1_ = storage_qmd.getPaddedSlotWidthBytes(state.agg_idx_)
                               ? storage_qmd.getPaddedSlotWidthBytes(state.agg_idx_)
                               : query_mem_desc_.getEffectiveKeyWidth();
    }
    for (size_t i = 0; i < nrows; ++i) {
      int8_t const* const pos_ptr = state.buf_ptr_ + i * state.compact_sz1_;
      int64_t pos = read_int_from_buff(pos_ptr, target_size);
      CHECK_GE(pos, 0);
      auto& frag_col_buffers = getColumnFrag(0, target_idx, pos);
      CHECK_LT(size_t(col_lazy_fetch.local_col_id), frag_col_buffers.size());
      int8_t const* const col_frag = frag_col_buffers[col_lazy_fetch.local_col_id];
      func(col_frag + pos * target_size);
    }
  } else {
    size_t const key_bytes_with_padding =
        align_to_int64(get_key_bytes_rowwise(storage_qmd));
    for (size_t i = 0; i < nrows; ++i) {
      int8_t const* const keys_ptr = row_ptr_rowwise(storage_->buff_, storage_qmd, i);
      int8_t const* const rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
      int64_t pos = *reinterpret_cast<int64_t const*>(rowwise_target_ptr);
      auto& frag_col_buffers = getColumnFrag(0, target_idx, pos);
      CHECK_LT(size_t(col_lazy_fetch.local_col_id), frag_col_buffers.size());
      int8_t const* const col_frag = frag_col_buffers[col_lazy_fetch.local_col_id];
      func(col_frag + pos * target_size);
    }
  }
}

namespace {

size_t get_truncated_row_count(size_t total_row_count, size_t limit, size_t offset) {
  if (total_row_count < offset) {
    return 0;
  }

  size_t total_truncated_row_count = total_row_count - offset;

  if (limit) {
    return std::min(total_truncated_row_count, limit);
  }

  return total_truncated_row_count;
}

}  // namespace

size_t ResultSet::rowCountImpl(const bool force_parallel) const {
  if (just_explain_) {
    return 1;
  }
  if (!permutation_.empty()) {
    // keep_first_ corresponds to SQL LIMIT
    // drop_first_ corresponds to SQL OFFSET
    return get_truncated_row_count(permutation_.size(), keep_first_, drop_first_);
  }
  if (!storage_) {
    return 0;
  }
  CHECK(permutation_.empty());
  if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    return binSearchRowCount();
  }

  constexpr size_t auto_parallel_row_count_threshold{20000UL};
  if (force_parallel || entryCount() >= auto_parallel_row_count_threshold) {
    return parallelRowCount();
  }
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  moveToBegin();
  size_t row_count{0};
  while (true) {
    auto crt_row = getNextRowUnlocked(false, false);
    if (crt_row.empty()) {
      break;
    }
    ++row_count;
  }
  moveToBegin();
  return row_count;
}

size_t ResultSet::rowCount(const bool force_parallel) const {
  // cached_row_count_ is atomic, so fetch it into a local variable first
  // to avoid repeat fetches
  const int64_t cached_row_count = cached_row_count_;
  if (cached_row_count != uninitialized_cached_row_count) {
    CHECK_GE(cached_row_count, 0);
    return cached_row_count;
  }
  setCachedRowCount(rowCountImpl(force_parallel));
  return cached_row_count_;
}

void ResultSet::invalidateCachedRowCount() const {
  cached_row_count_ = uninitialized_cached_row_count;
}

void ResultSet::setCachedRowCount(const size_t row_count) const {
  const int64_t signed_row_count = static_cast<int64_t>(row_count);
  const int64_t old_cached_row_count = cached_row_count_.exchange(signed_row_count);
  CHECK(old_cached_row_count == uninitialized_cached_row_count ||
        old_cached_row_count == signed_row_count);
}

size_t ResultSet::binSearchRowCount() const {
  if (!storage_) {
    return 0;
  }

  size_t row_count = storage_->binSearchRowCount();
  for (auto& s : appended_storage_) {
    row_count += s->binSearchRowCount();
  }

  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

size_t ResultSet::parallelRowCount() const {
  using namespace threading;
  auto execute_parallel_row_count = [this, query_id = logger::query_id()](
                                        const blocked_range<size_t>& r,
                                        size_t row_count) {
    auto qid_scope_guard = logger::set_thread_local_query_id(query_id);
    for (size_t i = r.begin(); i < r.end(); ++i) {
      if (!isRowAtEmpty(i)) {
        ++row_count;
      }
    }
    return row_count;
  };
  const auto row_count = parallel_reduce(blocked_range<size_t>(0, entryCount()),
                                         size_t(0),
                                         execute_parallel_row_count,
                                         std::plus<int>());
  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

bool ResultSet::isEmpty() const {
  // To simplify this function and de-dup logic with ResultSet::rowCount()
  // (mismatches between the two were causing bugs), we modified this function
  // to simply fetch rowCount(). The potential downside of this approach is that
  // in some cases more work will need to be done, as we can't just stop at the first row.
  // Mitigating that for most cases is the following:
  // 1) rowCount() is cached, so the logic for actually computing row counts will run only
  // once
  //    per result set.
  // 2) If the cache is empty (cached_row_count_ == -1), rowCount() will use parallel
  //    methods if deemed appropriate, which in many cases could be faster for a sparse
  //    large result set that single-threaded iteration from the beginning
  // 3) Often where isEmpty() is needed, rowCount() is also needed. Since the first call
  // to rowCount()
  //    will be cached, there is no extra overhead in these cases

  return rowCount() == size_t(0);
}

bool ResultSet::definitelyHasNoRows() const {
  return (!storage_ && !estimator_ && !just_explain_) || cached_row_count_ == 0;
}

const QueryMemoryDescriptor& ResultSet::getQueryMemDesc() const {
  return query_mem_desc_;
}

const std::vector<TargetInfo>& ResultSet::getTargetInfos() const {
  return targets_;
}

const std::vector<int64_t>& ResultSet::getTargetInitVals() const {
  CHECK(storage_);
  return storage_->target_init_vals_;
}

int8_t* ResultSet::getDeviceEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(device_estimator_buffer_);
  return device_estimator_buffer_->getMemoryPtr();
}

int8_t* ResultSet::getHostEstimatorBuffer() const {
  return host_estimator_buffer_;
}

void ResultSet::syncEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(!host_estimator_buffer_);
  CHECK_EQ(size_t(0), estimator_->getBufferSize() % sizeof(int64_t));
  host_estimator_buffer_ =
      static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  CHECK(device_estimator_buffer_);
  auto device_buffer_ptr = device_estimator_buffer_->getMemoryPtr();
  getBufferProvider()->copyFromDevice(
      host_estimator_buffer_, device_buffer_ptr, estimator_->getBufferSize(), device_id_);
}

void ResultSet::setQueueTime(const int64_t queue_time) {
  timings_.executor_queue_time = queue_time;
}

void ResultSet::setKernelQueueTime(const int64_t kernel_queue_time) {
  timings_.kernel_queue_time = kernel_queue_time;
}

void ResultSet::addCompilationQueueTime(const int64_t compilation_queue_time) {
  timings_.compilation_queue_time += compilation_queue_time;
}

int64_t ResultSet::getQueueTime() const {
  return timings_.executor_queue_time + timings_.kernel_queue_time +
         timings_.compilation_queue_time;
}

void ResultSet::moveToBegin() const {
  crt_row_buff_idx_ = 0;
  fetched_so_far_ = 0;
}

bool ResultSet::isTruncated() const {
  return keep_first_ + drop_first_;
}

bool ResultSet::isExplain() const {
  return just_explain_;
}

void ResultSet::setValidationOnlyRes() {
  for_validation_only_ = true;
}

bool ResultSet::isValidationOnlyRes() const {
  return for_validation_only_;
}

int ResultSet::getDeviceId() const {
  return device_id_;
}

QueryMemoryDescriptor ResultSet::fixupQueryMemoryDescriptor(
    const QueryMemoryDescriptor& query_mem_desc) {
  auto query_mem_desc_copy = query_mem_desc;
  query_mem_desc_copy.resetGroupColWidths(
      std::vector<int8_t>(query_mem_desc_copy.getGroupbyColCount(), 8));
  if (query_mem_desc.didOutputColumnar()) {
    return query_mem_desc_copy;
  }
  query_mem_desc_copy.alignPaddedSlots();
  return query_mem_desc_copy;
}

// Append non-empty indexes i in [begin,end) from findStorage(i) to permutation.
PermutationView ResultSet::initPermutationBuffer(PermutationView permutation,
                                                 PermutationIdx const begin,
                                                 PermutationIdx const end) const {
  auto timer = DEBUG_TIMER(__func__);
  for (PermutationIdx i = begin; i < end; ++i) {
    const auto storage_lookup_result = findStorage(i);
    const auto lhs_storage = storage_lookup_result.storage_ptr;
    const auto off = storage_lookup_result.fixedup_entry_idx;
    CHECK(lhs_storage);
    if (!lhs_storage->isEmptyEntry(off)) {
      permutation.push_back(i);
    }
  }
  return permutation;
}

const Permutation& ResultSet::getPermutationBuffer() const {
  return permutation_;
}

std::pair<size_t, size_t> ResultSet::getStorageIndex(const size_t entry_idx) const {
  size_t fixedup_entry_idx = entry_idx;
  auto entry_count = storage_->query_mem_desc_.getEntryCount();
  const bool is_rowwise_layout = !storage_->query_mem_desc_.didOutputColumnar();
  if (fixedup_entry_idx < entry_count) {
    return {0, fixedup_entry_idx};
  }
  fixedup_entry_idx -= entry_count;
  for (size_t i = 0; i < appended_storage_.size(); ++i) {
    const auto& desc = appended_storage_[i]->query_mem_desc_;
    CHECK_NE(is_rowwise_layout, desc.didOutputColumnar());
    entry_count = desc.getEntryCount();
    if (fixedup_entry_idx < entry_count) {
      return {i + 1, fixedup_entry_idx};
    }
    fixedup_entry_idx -= entry_count;
  }
  UNREACHABLE() << "entry_idx = " << entry_idx << ", query_mem_desc_.getEntryCount() = "
                << query_mem_desc_.getEntryCount();
  return {};
}

ResultSet::StorageLookupResult ResultSet::findStorage(const size_t entry_idx) const {
  auto [stg_idx, fixedup_entry_idx] = getStorageIndex(entry_idx);
  return {stg_idx ? appended_storage_[stg_idx - 1].get() : storage_.get(),
          fixedup_entry_idx,
          stg_idx};
}

double ResultSet::calculateQuantile(quantile::TDigest* const t_digest) {
  static_assert(sizeof(int64_t) == sizeof(quantile::TDigest*));
  CHECK(t_digest);
  t_digest->mergeBuffer();
  double const quantile = t_digest->quantile();
  return boost::math::isnan(quantile) ? NULL_DOUBLE : quantile;
}

size_t ResultSet::getLimit() const {
  return keep_first_;
}

size_t ResultSet::getOffset() const {
  return drop_first_;
}

const std::vector<std::string> ResultSet::getStringDictionaryPayloadCopy(
    const int dict_id) const {
  const auto sdp = row_set_mem_owner_->getOrAddStringDictProxy(dict_id);
  CHECK(sdp);
  return sdp->getDictionary()->copyStrings();
}

const std::pair<std::vector<int32_t>, std::vector<std::string>>
ResultSet::getUniqueStringsForDictEncodedTargetCol(const size_t col_idx) const {
  const auto col_type = colType(col_idx);
  CHECK(col_type->isExtDictionary());
  std::unordered_set<int32_t> unique_string_ids_set;
  const size_t num_entries = entryCount();
  std::vector<bool> targets_to_skip(colCount(), true);
  targets_to_skip[col_idx] = false;
  const auto null_val = inline_fixed_encoding_null_value(col_type);

  for (size_t row_idx = 0; row_idx < num_entries; ++row_idx) {
    const auto result_row = getRowAtNoTranslations(row_idx, targets_to_skip);
    if (!result_row.empty()) {
      const auto scalar_col_val = boost::get<ScalarTargetValue>(result_row[col_idx]);
      const int32_t string_id = static_cast<int32_t>(boost::get<int64_t>(scalar_col_val));
      if (string_id != null_val) {
        unique_string_ids_set.emplace(string_id);
      }
    }
  }

  const size_t num_unique_strings = unique_string_ids_set.size();
  std::vector<int32_t> unique_string_ids(num_unique_strings);
  size_t string_idx{0};
  for (const auto unique_string_id : unique_string_ids_set) {
    unique_string_ids[string_idx++] = unique_string_id;
  }

  const int32_t dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  const auto sdp = row_set_mem_owner_->getOrAddStringDictProxy(dict_id);
  CHECK(sdp);

  return std::make_pair(unique_string_ids, sdp->getStrings(unique_string_ids));
}

/**
 * Determines if it is possible to directly form a ColumnarResults class from this
 * result set, bypassing the default columnarization.
 *
 * NOTE: If there exists a permutation vector (i.e., in some ORDER BY queries), it
 * becomes equivalent to the row-wise columnarization.
 */
bool ResultSet::isDirectColumnarConversionPossible() const {
  if (query_mem_desc_.didOutputColumnar()) {
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::Projection ||
                                    (query_mem_desc_.getQueryDescriptionType() ==
                                         QueryDescriptionType::GroupByPerfectHash ||
                                     query_mem_desc_.getQueryDescriptionType() ==
                                         QueryDescriptionType::GroupByBaselineHash));
  } else {
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByPerfectHash ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByBaselineHash);
  }
}

bool ResultSet::isZeroCopyColumnarConversionPossible(size_t column_idx) const {
  return query_mem_desc_.didOutputColumnar() &&
         query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection &&
         !colType(column_idx)->isVarLen() && !colType(column_idx)->isArray() &&
         appended_storage_.empty() && storage_ &&
         (lazy_fetch_info_.empty() || !lazy_fetch_info_[column_idx].is_lazily_fetched);
}

bool ResultSet::isChunkedZeroCopyColumnarConversionPossible(size_t column_idx) const {
  return query_mem_desc_.didOutputColumnar() &&
         query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection &&
         !colType(column_idx)->isVarLen() && !colType(column_idx)->isArray() &&
         storage_ &&
         (lazy_fetch_info_.empty() || !lazy_fetch_info_[column_idx].is_lazily_fetched);
}

const int8_t* ResultSet::getColumnarBuffer(size_t column_idx) const {
  CHECK(isZeroCopyColumnarConversionPossible(column_idx));
  size_t slot_idx = query_mem_desc_.getSlotIndexForSingleSlotCol(column_idx);
  return storage_->getUnderlyingBuffer() + query_mem_desc_.getColOffInBytes(slot_idx);
}

std::vector<std::pair<const int8_t*, size_t>> ResultSet::getChunkedColumnarBuffer(
    size_t column_idx) const {
  CHECK(isChunkedZeroCopyColumnarConversionPossible(column_idx));

  std::vector<std::pair<const int8_t*, size_t>> retval;
  retval.reserve(1 + appended_storage_.size());

  size_t current_storage_rows = storage_->binSearchRowCount();
  size_t rows_to_skip = drop_first_;
  // RowCount value should be cached and take into account size, limit and offset
  size_t rows_to_fetch = rowCount();
  size_t slot_idx = query_mem_desc_.getSlotIndexForSingleSlotCol(column_idx);

  if (current_storage_rows <= rows_to_skip) {
    rows_to_skip -= current_storage_rows;
  } else {
    size_t fetch_from_current_storage =
        std::min(current_storage_rows - rows_to_skip, rows_to_fetch);
    retval.emplace_back(storage_->getUnderlyingBuffer() +
                            storage_->getColOffInBytes(slot_idx) +
                            colType(column_idx)->size() * rows_to_skip,
                        fetch_from_current_storage);
    rows_to_fetch -= fetch_from_current_storage;
    rows_to_skip = 0;
  }

  for (auto& storage_uptr : appended_storage_) {
    if (rows_to_fetch == 0) {
      break;
    }
    const int8_t* ptr =
        storage_uptr->getUnderlyingBuffer() + storage_uptr->getColOffInBytes(slot_idx);
    current_storage_rows = storage_uptr->binSearchRowCount();
    if (current_storage_rows <= rows_to_skip) {
      rows_to_skip -= current_storage_rows;
    } else {
      size_t fetch_from_current_storage =
          std::min(current_storage_rows - rows_to_skip, rows_to_fetch);
      retval.emplace_back(ptr + colType(column_idx)->size() * rows_to_skip,
                          fetch_from_current_storage);
      rows_to_fetch -= fetch_from_current_storage;
      rows_to_skip = 0;
    }
  }
  return retval;
}

size_t ResultSet::computeVarLenOffsets(size_t col_idx, int32_t* offsets) const {
  auto type = colType(col_idx);
  CHECK(type->isVarLen());
  size_t arr_elem_size =
      type->isVarLenArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType()->size() : 1;
  bool lazy_fetch =
      !lazy_fetch_info_.empty() && lazy_fetch_info_[col_idx].is_lazily_fetched;

  size_t data_slot_idx = 0;
  size_t data_slot_offs = 0;
  size_t size_slot_idx = 0;
  size_t size_slot_offs = 0;
  // Compute required slot index.
  for (size_t i = 0; i < col_idx; ++i) {
    // slot offset in a row is computed for rowwise access.
    if (!query_mem_desc_.didOutputColumnar()) {
      data_slot_offs = advance_target_ptr_row_wise(data_slot_offs,
                                                   targets_[i],
                                                   data_slot_idx,
                                                   query_mem_desc_,
                                                   separate_varlen_storage_valid_);
    }
    data_slot_idx =
        advance_slot(data_slot_idx, targets_[i], separate_varlen_storage_valid_);
  }
  if (!separate_varlen_storage_valid_ && !lazy_fetch) {
    size_slot_offs =
        data_slot_offs + query_mem_desc_.getPaddedSlotWidthBytes(data_slot_idx);
    size_slot_idx = data_slot_idx + 1;
  } else {
    size_slot_idx = data_slot_idx;
    size_slot_offs = data_slot_offs;
  }

  // Translate varlen value to its length. Return -1 for NULLs.
  auto slot_val_to_length = [this, lazy_fetch, col_idx, type](
                                size_t storage_idx,
                                int64_t val,
                                const int8_t* size_slot_ptr,
                                size_t size_slot_sz) -> int32_t {
    if (separate_varlen_storage_valid_ && !targets_[col_idx].is_agg) {
      if (val >= 0) {
        const auto& varlen_buffer_for_storage = serialized_varlen_buffer_[storage_idx];
        return varlen_buffer_for_storage[val].size();
      }
      return -1;
    }

    if (lazy_fetch) {
      auto& frag_col_buffers = getColumnFrag(storage_idx, col_idx, val);
      bool is_end{false};
      if (type->isString()) {
        VarlenDatum vd;
        ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(
                              frag_col_buffers[lazy_fetch_info_[col_idx].local_col_id])),
                          val,
                          false,
                          &vd,
                          &is_end);
        CHECK(!is_end);
        return vd.is_null ? -1 : vd.length;
      } else {
        ArrayDatum ad;
        ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(
                              frag_col_buffers[lazy_fetch_info_[col_idx].local_col_id])),
                          val,
                          &ad,
                          &is_end);
        CHECK(!is_end);
        return ad.is_null ? -1 : ad.length;
      }
    }

    if (val)
      return read_int_from_buff(size_slot_ptr, size_slot_sz);
    return -1;
  };

  offsets[0] = 0;
  size_t row_idx = 0;
  ResultSetRowIterator iter(this);
  ++iter;
  const auto data_elem_size = query_mem_desc_.getPaddedSlotWidthBytes(data_slot_idx);
  const auto size_elem_size = query_mem_desc_.getPaddedSlotWidthBytes(size_slot_idx);
  while (iter.global_entry_idx_valid_) {
    const auto storage_lookup_result = findStorage(iter.global_entry_idx_);
    auto storage = storage_lookup_result.storage_ptr;
    auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;

    const int8_t* elem_ptr = nullptr;
    const int8_t* size_ptr = nullptr;
    if (query_mem_desc_.didOutputColumnar()) {
      auto col_ptr =
          storage->buff_ + storage->query_mem_desc_.getColOffInBytes(data_slot_idx);
      elem_ptr = col_ptr + data_elem_size * local_entry_idx;
      auto size_col_ptr =
          storage->buff_ + storage->query_mem_desc_.getColOffInBytes(size_slot_idx);
      size_ptr = size_col_ptr + size_elem_size * local_entry_idx;
    } else {
      auto keys_ptr = row_ptr_rowwise(storage->buff_, query_mem_desc_, local_entry_idx);
      const auto key_bytes_with_padding =
          align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
      elem_ptr = keys_ptr + key_bytes_with_padding + data_slot_offs;
      size_ptr = keys_ptr + key_bytes_with_padding + size_slot_offs;
    }

    auto val = read_int_from_buff(elem_ptr, data_elem_size);
    auto elem_length = slot_val_to_length(
        storage_lookup_result.storage_idx, val, size_ptr, size_elem_size);
    if (elem_length < 0) {
      if (type->isString()) {
        offsets[row_idx + 1] = offsets[row_idx];
      } else {
        offsets[row_idx + 1] = -std::abs(offsets[row_idx]);
      }
    } else {
      offsets[row_idx + 1] = std::abs(offsets[row_idx]) + elem_length * arr_elem_size;
    }

    ++iter;
    ++row_idx;
  }

  return row_idx + 1;
}

// Returns a bitmap (and total number) of all single slot targets
std::tuple<std::vector<bool>, size_t> ResultSet::getSingleSlotTargetBitmap() const {
  std::vector<bool> target_bitmap(targets_.size(), true);
  size_t num_single_slot_targets = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    auto sql_type = targets_[target_idx].type;
    if (targets_[target_idx].is_agg &&
        targets_[target_idx].agg_kind == hdk::ir::AggType::kAvg) {
      target_bitmap[target_idx] = false;
    } else if (sql_type->isString() || sql_type->isArray()) {
      target_bitmap[target_idx] = false;
    } else {
      num_single_slot_targets++;
    }
  }
  return std::make_tuple(std::move(target_bitmap), num_single_slot_targets);
}

/**
 * This function returns a bitmap and population count of it, where it denotes
 * all supported single-column targets suitable for direct columnarization.
 *
 * The final goal is to remove the need for such selection, but at the moment for any
 * target that doesn't qualify for direct columnarization, we use the traditional
 * result set's iteration to handle it (e.g., count distinct, approximate count
 * distinct)
 */
std::tuple<std::vector<bool>, size_t> ResultSet::getSupportedSingleSlotTargetBitmap()
    const {
  CHECK(isDirectColumnarConversionPossible());
  auto [single_slot_targets, num_single_slot_targets] = getSingleSlotTargetBitmap();

  for (size_t target_idx = 0; target_idx < single_slot_targets.size(); target_idx++) {
    const auto& target = targets_[target_idx];
    if (single_slot_targets[target_idx] &&
        (is_distinct_target(target) ||
         target.agg_kind == hdk::ir::AggType::kApproxQuantile ||
         (target.is_agg && target.agg_kind == hdk::ir::AggType::kSample &&
          target.type->isFp32()))) {
      single_slot_targets[target_idx] = false;
      num_single_slot_targets--;
    }
  }
  CHECK_GE(num_single_slot_targets, size_t(0));
  return std::make_tuple(std::move(single_slot_targets), num_single_slot_targets);
}

// Returns the starting slot index for all targets in the result set
std::vector<size_t> ResultSet::getSlotIndicesForTargetIndices() const {
  std::vector<size_t> slot_indices(targets_.size(), 0);
  size_t slot_index = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    slot_indices[target_idx] = slot_index;
    slot_index = advance_slot(slot_index, targets_[target_idx], false);
  }
  return slot_indices;
}

size_t ResultSet::getNDVEstimator() const {
  CHECK(dynamic_cast<const hdk::ir::NDVEstimator*>(estimator_.get()));
  CHECK(host_estimator_buffer_);
  auto bits_set = bitmap_set_size(host_estimator_buffer_, estimator_->getBufferSize());
  if (bits_set == 0) {
    // empty result set, return 1 for a groups buffer size of 1
    return 1;
  }
  const auto total_bits = estimator_->getBufferSize() * 8;
  CHECK_LE(bits_set, total_bits);
  const auto unset_bits = total_bits - bits_set;
  const auto ratio = static_cast<double>(unset_bits) / total_bits;
  if (ratio == 0.) {
    LOG(WARNING)
        << "Failed to get a high quality cardinality estimation, falling back to "
           "approximate group by buffer size guess.";
    return 0;
  }
  return -static_cast<double>(total_bits) * log(ratio);
}

void ResultSet::initializeStorage() const {
  if (query_mem_desc_.didOutputColumnar()) {
    storage_->initializeColWise();
  } else {
    storage_->initializeRowWise();
  }
}

// namespace result_set

bool result_set::can_use_parallel_algorithms(const ResultSet& rows) {
  return !rows.isTruncated();
}

bool result_set::use_parallel_algorithms(const ResultSet& rows) {
  return result_set::can_use_parallel_algorithms(rows) && rows.entryCount() >= 20000;
}

BufferProvider* ResultSet::getBufferProvider() const {
  return data_mgr_ ? data_mgr_->getBufferProvider() : nullptr;
}
