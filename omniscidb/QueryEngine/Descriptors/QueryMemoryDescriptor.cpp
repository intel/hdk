/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "QueryEngine/Descriptors/QueryMemoryDescriptor.h"

#include <boost/algorithm/string.hpp>

#include "DataMgr/DataMgr.h"
#include "QueryEngine/ColRangeInfo.h"
#include "QueryEngine/StreamingTopN.h"

QueryMemoryDescriptor::QueryMemoryDescriptor(
    Data_Namespace::DataMgr* data_mgr,
    ConfigPtr config,
    const std::vector<InputTableInfo>& query_infos,
    const bool use_bump_allocator,
    const bool approx_quantile,
    const bool allow_multifrag,
    const bool keyless_hash,
    const bool interleaved_bins_on_gpu,
    const int32_t idx_target_as_key,
    const ColRangeInfo& col_range_info,
    const ColSlotContext& col_slot_context,
    const std::vector<int8_t>& group_col_widths,
    const int8_t group_col_compact_width,
    const std::vector<int64_t>& target_groupby_indices,
    const size_t entry_count,
    const CountDistinctDescriptors count_distinct_descriptors,
    const bool sort_on_gpu_hint,
    const bool output_columnar_hint,
    const bool must_use_baseline_sort,
    const bool use_streaming_top_n)
    : data_mgr_(data_mgr)
    , config_(config)
    , query_desc_type_(col_range_info.hash_type_)
    , keyless_hash_(keyless_hash)
    , interleaved_bins_on_gpu_(interleaved_bins_on_gpu)
    , idx_target_as_key_(idx_target_as_key)
    , group_col_widths_(group_col_widths)
    , group_col_compact_width_(group_col_compact_width)
    , target_groupby_indices_(target_groupby_indices)
    , entry_count_(entry_count)
    , min_val_(col_range_info.min)
    , max_val_(col_range_info.max)
    , bucket_(col_range_info.bucket)
    , has_nulls_(col_range_info.has_nulls)
    , count_distinct_descriptors_(count_distinct_descriptors)
    , output_columnar_(false)
    , must_use_baseline_sort_(must_use_baseline_sort)
    , is_table_function_(false)
    , use_streaming_top_n_(use_streaming_top_n)
    , force_4byte_float_(false)
    , col_slot_context_(col_slot_context) {
  col_slot_context_.setAllUnsetSlotsPaddedSize(8);
  col_slot_context_.validate();

  sort_on_gpu_ = sort_on_gpu_hint && canOutputColumnar() && !keyless_hash_;

  if (sort_on_gpu_) {
    CHECK(!use_bump_allocator);
    output_columnar_ = true;
  } else {
    switch (query_desc_type_) {
      case QueryDescriptionType::Projection:
        output_columnar_ = output_columnar_hint;
        break;
      case QueryDescriptionType::GroupByPerfectHash:
      case QueryDescriptionType::GroupByBaselineHash:
      case QueryDescriptionType::NonGroupedAggregate:
        output_columnar_ = output_columnar_hint &&
                           QueryMemoryDescriptor::countDescriptorsLogicallyEmpty(
                               count_distinct_descriptors_) &&
                           !approx_quantile;
        break;
      default:
        output_columnar_ = false;
        break;
    }
  }

  if (isLogicalSizedColumnsAllowed()) {
    // TODO(adb): Ensure fixed size buffer allocations are correct with all logical
    // column sizes
    CHECK(!use_bump_allocator);
    col_slot_context_.setAllSlotsPaddedSizeToLogicalSize();
    col_slot_context_.validate();
  }

#ifdef HAVE_CUDA
  // Check Streaming Top N heap usage, bail if > max slab size, CUDA ONLY
  if (use_streaming_top_n_ && data_mgr_->gpusPresent()) {
    const auto thread_count = blockSize() * gridSize();
    const auto total_buff_size =
        streaming_top_n::get_heap_size(getRowSize(), getEntryCount(), thread_count);
    if (total_buff_size > config_->mem.gpu.max_slab_size) {
      throw StreamingTopNOOM(total_buff_size);
    }
  }
#endif
}

QueryMemoryDescriptor::QueryMemoryDescriptor()
    : data_mgr_(nullptr)
    , config_(nullptr)
    , query_desc_type_(QueryDescriptionType::Projection)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_compact_width_(0)
    , entry_count_(0)
    , min_val_(0)
    , max_val_(0)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(false)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

QueryMemoryDescriptor::QueryMemoryDescriptor(Data_Namespace::DataMgr* data_mgr,
                                             ConfigPtr config,
                                             const size_t entry_count,
                                             const QueryDescriptionType query_desc_type,
                                             const bool is_table_function)
    : data_mgr_(data_mgr)
    , config_(config)
    , query_desc_type_(query_desc_type)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_compact_width_(0)
    , entry_count_(entry_count)
    , min_val_(0)
    , max_val_(0)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(is_table_function)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

QueryMemoryDescriptor::QueryMemoryDescriptor(const QueryDescriptionType query_desc_type,
                                             const int64_t min_val,
                                             const int64_t max_val,
                                             const bool has_nulls,
                                             const std::vector<int8_t>& group_col_widths)
    : data_mgr_(nullptr)
    , config_(nullptr)
    , query_desc_type_(query_desc_type)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_widths_(group_col_widths)
    , group_col_compact_width_(0)
    , entry_count_(0)
    , min_val_(min_val)
    , max_val_(max_val)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(false)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

bool QueryMemoryDescriptor::operator==(const QueryMemoryDescriptor& other) const {
  // Note that this method does not check ptr reference members (e.g. data_mgr_) or
  // entry_count_
  if (query_desc_type_ != other.query_desc_type_) {
    return false;
  }
  if (keyless_hash_ != other.keyless_hash_) {
    return false;
  }
  if (interleaved_bins_on_gpu_ != other.interleaved_bins_on_gpu_) {
    return false;
  }
  if (idx_target_as_key_ != other.idx_target_as_key_) {
    return false;
  }
  if (force_4byte_float_ != other.force_4byte_float_) {
    return false;
  }
  if (group_col_widths_ != other.group_col_widths_) {
    return false;
  }
  if (group_col_compact_width_ != other.group_col_compact_width_) {
    return false;
  }
  if (target_groupby_indices_ != other.target_groupby_indices_) {
    return false;
  }
  if (min_val_ != other.min_val_) {
    return false;
  }
  if (max_val_ != other.max_val_) {
    return false;
  }
  if (bucket_ != other.bucket_) {
    return false;
  }
  if (has_nulls_ != other.has_nulls_) {
    return false;
  }
  if (count_distinct_descriptors_.size() != count_distinct_descriptors_.size()) {
    return false;
  } else {
    // Count distinct descriptors can legitimately differ in device only.
    for (size_t i = 0; i < count_distinct_descriptors_.size(); ++i) {
      auto ref_count_distinct_desc = other.count_distinct_descriptors_[i];
      auto count_distinct_desc = count_distinct_descriptors_[i];
      count_distinct_desc.device_type = ref_count_distinct_desc.device_type;
      if (ref_count_distinct_desc != count_distinct_desc) {
        return false;
      }
    }
  }
  if (sort_on_gpu_ != other.sort_on_gpu_) {
    return false;
  }
  if (output_columnar_ != other.output_columnar_) {
    return false;
  }
  if (col_slot_context_ != other.col_slot_context_) {
    return false;
  }
  return true;
}

size_t QueryMemoryDescriptor::getColsSize() const {
  return col_slot_context_.getAllSlotsAlignedPaddedSize();
}

size_t QueryMemoryDescriptor::getRowSize() const {
  CHECK(!output_columnar_);
  size_t total_bytes{0};
  if (keyless_hash_) {
    // ignore, there's no group column in the output buffer
    CHECK(query_desc_type_ == QueryDescriptionType::GroupByPerfectHash);
  } else {
    total_bytes += group_col_widths_.size() * getEffectiveKeyWidth();
    total_bytes = align_to_int64(total_bytes);
  }
  total_bytes += getColsSize();
  return align_to_int64(total_bytes);
}

size_t QueryMemoryDescriptor::getWarpCount() const {
  return (interleaved_bins_on_gpu_ ? warpSize() : 1);
}

size_t QueryMemoryDescriptor::getCompactByteWidth() const {
  return col_slot_context_.getCompactByteWidth();
}

/**
 * Returns the maximum total number of bytes (including required paddings) to store
 * all non-lazy columns' results for columnar cases.
 *
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers() const {
  CHECK(output_columnar_);
  return col_slot_context_.getTotalBytesOfColumnarBuffers(entry_count_);
}

/**
 * This is a helper function that returns the total number of bytes (including
 * required paddings) to store all non-lazy columns' results for columnar cases.
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers(
    const size_t num_entries_per_column) const {
  return col_slot_context_.getTotalBytesOfColumnarBuffers(num_entries_per_column);
}

/**
 * Returns the effective total number of bytes from columnar projections, which
 * includes 1) total number of bytes used to store all non-lazy columns 2) total
 * number of bytes used to store row indices (for lazy fetches, etc.)
 *
 * NOTE: this function does not represent the buffer sizes dedicated for the results,
 * but the required memory to fill all valid results into a compact new buffer (with
 * no holes in it)
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarProjections(
    const size_t projection_count) const {
  constexpr size_t row_index_width = sizeof(int64_t);
  return getTotalBytesOfColumnarBuffers(projection_count) +
         row_index_width * projection_count;
}

size_t QueryMemoryDescriptor::getColOnlyOffInBytes(const size_t col_idx) const {
  return col_slot_context_.getColOnlyOffInBytes(col_idx);
}

/*
 * Returns the memory offset in bytes for a specific agg column in the output
 * memory buffer. Depending on the query type, there may be some extra portion
 * of memory prepended at the beginning of the buffer. A brief description of
 * the memory layout is as follows:
 * 1. projections: index column (64bit) + all target columns
 * 2. group by: all group columns (64-bit each) + all agg columns
 * 2a. if keyless, there is no prepending group column stored at the beginning
 */
size_t QueryMemoryDescriptor::getColOffInBytes(const size_t col_idx) const {
  const auto warp_count = getWarpCount();
  if (output_columnar_) {
    CHECK_EQ(size_t(1), warp_count);
    size_t offset{0};
    if (!keyless_hash_) {
      offset += getPrependedGroupBufferSizeInBytes();
    }
    for (size_t index = 0; index < col_idx; ++index) {
      offset += align_to_int64(getPaddedSlotWidthBytes(index) * entry_count_);
    }
    return offset;
  }

  size_t offset{0};
  if (keyless_hash_) {
    // ignore, there's no group column in the output buffer
    CHECK(query_desc_type_ == QueryDescriptionType::GroupByPerfectHash);
  } else {
    offset += group_col_widths_.size() * getEffectiveKeyWidth();
    offset = align_to_int64(offset);
  }
  offset += getColOnlyOffInBytes(col_idx);
  return offset;
}

/*
 * Returns the memory offset for a particular group column in the prepended group
 * columns portion of the memory.
 */
size_t QueryMemoryDescriptor::getPrependedGroupColOffInBytes(
    const size_t group_idx) const {
  CHECK(output_columnar_);
  CHECK(group_idx < getGroupbyColCount());
  size_t offset{0};
  for (size_t col_idx = 0; col_idx < group_idx; col_idx++) {
    // TODO(Saman): relax that int64_bit part immediately
    offset += align_to_int64(
        std::max(groupColWidth(col_idx), static_cast<int8_t>(sizeof(int64_t))) *
        getEntryCount());
  }
  return offset;
}

/*
 * Returns total amount of memory prepended at the beginning of the output memory
 * buffer.
 */
size_t QueryMemoryDescriptor::getPrependedGroupBufferSizeInBytes() const {
  CHECK(output_columnar_);
  size_t buffer_size{0};
  for (size_t group_idx = 0; group_idx < getGroupbyColCount(); group_idx++) {
    buffer_size += align_to_int64(
        std::max(groupColWidth(group_idx), static_cast<int8_t>(sizeof(int64_t))) *
        getEntryCount());
  }
  return buffer_size;
}

size_t QueryMemoryDescriptor::getColOffInBytesInNextBin(const size_t col_idx) const {
  auto warp_count = getWarpCount();
  if (output_columnar_) {
    CHECK_EQ(size_t(1), group_col_widths_.size());
    CHECK_EQ(size_t(1), warp_count);
    return getPaddedSlotWidthBytes(col_idx);
  }

  return warp_count * getRowSize();
}

size_t QueryMemoryDescriptor::getNextColOffInBytes(const int8_t* col_ptr,
                                                   const size_t bin,
                                                   const size_t col_idx) const {
  CHECK(!output_columnar_ || bin < entry_count_);
  size_t offset{0};
  auto warp_count = getWarpCount();
  const auto chosen_bytes = getPaddedSlotWidthBytes(col_idx);
  const auto total_slot_count = getSlotCount();
  if (col_idx + 1 == total_slot_count) {
    if (output_columnar_) {
      return (entry_count_ - bin) * chosen_bytes;
    } else {
      return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
    }
  }

  const auto next_chosen_bytes = getPaddedSlotWidthBytes(col_idx + 1);
  if (output_columnar_) {
    CHECK_EQ(size_t(1), group_col_widths_.size());
    CHECK_EQ(size_t(1), warp_count);

    offset = align_to_int64(entry_count_ * chosen_bytes);

    offset += bin * (next_chosen_bytes - chosen_bytes);
    return offset;
  }

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getNextColOffInBytesRowOnly(const int8_t* col_ptr,
                                                          const size_t col_idx) const {
  const auto chosen_bytes = getPaddedSlotWidthBytes(col_idx);
  const auto total_slot_count = getSlotCount();
  if (col_idx + 1 == total_slot_count) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  }

  const auto next_chosen_bytes = getPaddedSlotWidthBytes(col_idx + 1);

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(
    const size_t max_rows,
    const unsigned thread_count,
    const ExecutorDeviceType device_type) const {
  if (use_streaming_top_n_) {
    CHECK_GT(max_rows, size_t(0));
    return streaming_top_n::get_heap_size(getRowSize(), max_rows, thread_count);
  }
  return getBufferSizeBytes(device_type, entry_count_);
}

/**
 * Returns total amount of output buffer memory for each device (CPU/GPU)
 *
 * Columnar:
 *  if projection: it returns index buffer + columnar buffer (all non-lazy columns)
 *  if group by: it returns the amount required for each group column (assumes 64-bit
 * per group) + columnar buffer (all involved agg columns)
 *
 * Row-wise:
 *  returns required memory per row multiplied by number of entries
 */
size_t QueryMemoryDescriptor::getBufferSizeBytes(const ExecutorDeviceType device_type,
                                                 const size_t entry_count) const {
  if (keyless_hash_ && !output_columnar_) {
    CHECK_GE(group_col_widths_.size(), size_t(1));
    auto row_bytes = align_to_int64(getColsSize());

    return (interleavedBins(device_type) ? warpSize() : 1) * entry_count * row_bytes;
  }

  constexpr size_t row_index_width = sizeof(int64_t);
  size_t total_bytes{0};
  if (output_columnar_) {
    total_bytes = (query_desc_type_ == QueryDescriptionType::Projection
                       ? row_index_width * entry_count
                       : sizeof(int64_t) * group_col_widths_.size() * entry_count) +
                  getTotalBytesOfColumnarBuffers();
  } else {
    total_bytes = getRowSize() * entry_count;
  }

  return total_bytes;
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(
    const ExecutorDeviceType device_type) const {
  return getBufferSizeBytes(device_type, entry_count_);
}

void QueryMemoryDescriptor::setOutputColumnar(const bool val) {
  output_columnar_ = val;
  if (isLogicalSizedColumnsAllowed()) {
    col_slot_context_.setAllSlotsPaddedSizeToLogicalSize();
  }
}

/*
 * Indicates the query types that are currently allowed to use the logical
 * sized columns instead of padded sized ones.
 */
bool QueryMemoryDescriptor::isLogicalSizedColumnsAllowed() const {
  // In distributed mode, result sets are serialized using rowwise iterators, so we use
  // consistent slot widths for now
  return output_columnar_ && (query_desc_type_ == QueryDescriptionType::Projection);
}

size_t QueryMemoryDescriptor::getBufferColSlotCount() const {
  size_t total_slot_count = col_slot_context_.getSlotCount();

  if (target_groupby_indices_.empty()) {
    return total_slot_count;
  }
  return total_slot_count - std::count_if(target_groupby_indices_.begin(),
                                          target_groupby_indices_.end(),
                                          [](const int64_t i) { return i >= 0; });
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
          getGroupbyColCount() == 1);
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return query_desc_type_ != QueryDescriptionType::NonGroupedAggregate;
}

bool QueryMemoryDescriptor::blocksShareMemory() const {
  if (is_table_function_) {
    return true;
  }
  if (!countDescriptorsLogicallyEmpty(count_distinct_descriptors_)) {
    return true;
  }
  CHECK(data_mgr_);
  if (!data_mgr_->getCudaMgr() ||
      query_desc_type_ == QueryDescriptionType::GroupByBaselineHash ||
      query_desc_type_ == QueryDescriptionType::Projection ||
      (query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
       getGroupbyColCount() > 1)) {
    return true;
  }
  return query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
         many_entries(max_val_, min_val_, bucket_);
}

bool QueryMemoryDescriptor::lazyInitGroups(const ExecutorDeviceType device_type) const {
#ifdef HAVE_L0
  return false; /* L0 path currently does not provide lazy group by buffers
                 * initialization
                 */
#else
  return device_type == ExecutorDeviceType::GPU &&
         countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
#endif
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu_ && device_type == ExecutorDeviceType::GPU;
}

// TODO(Saman): an implementation detail, so move this out of QMD
bool QueryMemoryDescriptor::isWarpSyncRequired(
    const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::GPU &&
      data_mgr_->getGpuMgr()->getPlatform() == GpuMgrPlatform::CUDA) {
    return data_mgr_->getCudaMgr()->isArchVoltaOrGreaterForAll();
  }
  return false;
}

size_t QueryMemoryDescriptor::getColCount() const {
  return col_slot_context_.getColCount();
}

size_t QueryMemoryDescriptor::getSlotCount() const {
  return col_slot_context_.getSlotCount();
}

const int8_t QueryMemoryDescriptor::getPaddedSlotWidthBytes(const size_t slot_idx) const {
  return col_slot_context_.getSlotInfo(slot_idx).padded_size;
}

void QueryMemoryDescriptor::setPaddedSlotWidthBytes(const size_t slot_idx,
                                                    const int8_t bytes) {
  col_slot_context_.setPaddedSlotWidthBytes(slot_idx, bytes);
}

const int8_t QueryMemoryDescriptor::getLogicalSlotWidthBytes(
    const size_t slot_idx) const {
  return col_slot_context_.getSlotInfo(slot_idx).logical_size;
}

const size_t QueryMemoryDescriptor::getSlotIndexForSingleSlotCol(
    const size_t col_idx) const {
  const auto& col_slots = col_slot_context_.getSlotsForCol(col_idx);
  CHECK_EQ(col_slots.size(), size_t(1));
  return col_slots.front();
}

void QueryMemoryDescriptor::useConsistentSlotWidthSize(const int8_t slot_width_size) {
  col_slot_context_.setAllSlotsSize(slot_width_size);
}

size_t QueryMemoryDescriptor::getRowWidth() const {
  // Note: Actual row size may include padding (see ResultSetBufferAccessors.h)
  return col_slot_context_.getAllSlotsPaddedSize();
}

int8_t QueryMemoryDescriptor::updateActualMinByteWidth(
    const int8_t actual_min_byte_width) const {
  return col_slot_context_.getMinPaddedByteSize(actual_min_byte_width);
}

void QueryMemoryDescriptor::addColSlotInfo(
    const std::vector<std::tuple<int8_t, int8_t>>& slots_for_col) {
  col_slot_context_.addColumn(slots_for_col);
}

void QueryMemoryDescriptor::clearSlotInfo() {
  col_slot_context_.clear();
}

void QueryMemoryDescriptor::alignPaddedSlots() {
  col_slot_context_.alignPaddedSlots(sortOnGpu());
}

bool QueryMemoryDescriptor::canOutputColumnar() const {
  return usesGetGroupValueFast() && threadsShareMemory() && blocksShareMemory() &&
         !interleavedBins(ExecutorDeviceType::GPU) &&
         countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
}

std::string QueryMemoryDescriptor::queryDescTypeToString() const {
  switch (query_desc_type_) {
    case QueryDescriptionType::GroupByPerfectHash:
      return "Perfect Hash";
    case QueryDescriptionType::GroupByBaselineHash:
      return "Baseline Hash";
    case QueryDescriptionType::Projection:
      return "Projection";
    case QueryDescriptionType::NonGroupedAggregate:
      return "Non-grouped Aggregate";
    case QueryDescriptionType::Estimator:
      return "Estimator";
    default:
      UNREACHABLE();
  }
  return "";
}

std::string QueryMemoryDescriptor::toString() const {
  auto str = reductionKey();
  str += "\tInterleaved Bins on GPU: " + ::toString(interleaved_bins_on_gpu_) + "\n";
  str += "\tBlocks Share Memory: " + ::toString(blocksShareMemory()) + "\n";
  str += "\tThreads Share Memory: " + ::toString(threadsShareMemory()) + "\n";
  str += "\tUses Fast Group Values: " + ::toString(usesGetGroupValueFast()) + "\n";
  str +=
      "\tLazy Init Groups (GPU): " + ::toString(lazyInitGroups(ExecutorDeviceType::GPU)) +
      "\n";
  str += "\tEntry Count: " + std::to_string(entry_count_) + "\n";
  str += "\tMin Val (perfect hash only): " + std::to_string(min_val_) + "\n";
  str += "\tMax Val (perfect hash only): " + std::to_string(max_val_) + "\n";
  str += "\tBucket Val (perfect hash only): " + std::to_string(bucket_) + "\n";
  str += "\tSort on GPU: " + ::toString(sort_on_gpu_) + "\n";
  str += "\tUse Streaming Top N: " + ::toString(use_streaming_top_n_) + "\n";
  str += "\tOutput Columnar: " + ::toString(output_columnar_) + "\n";
  str += "\tUse Baseline Sort: " + ::toString(must_use_baseline_sort_) + "\n";
  str += "\tIs Table Function: " + ::toString(is_table_function_) + "\n";
  return str;
}

std::string QueryMemoryDescriptor::reductionKey() const {
  std::string str;
  str += "Query Memory Descriptor State\n";
  str += "\tQuery Type: " + queryDescTypeToString() + "\n";
  str +=
      "\tKeyless Hash: " + ::toString(keyless_hash_) +
      (keyless_hash_ ? ", target index for key: " + std::to_string(getTargetIdxForKey())
                     : "") +
      "\n";
  str += "\tEffective key width: " + std::to_string(getEffectiveKeyWidth()) + "\n";
  str += "\tNumber of group columns: " + std::to_string(getGroupbyColCount()) + "\n";
  const auto group_indices_size = targetGroupbyIndicesSize();
  if (group_indices_size) {
    std::vector<std::string> group_indices_strings;
    for (size_t target_idx = 0; target_idx < group_indices_size; ++target_idx) {
      group_indices_strings.push_back(std::to_string(getTargetGroupbyIndex(target_idx)));
    }
    str += "\tTarget group by indices: " +
           boost::algorithm::join(group_indices_strings, ",") + "\n";
  }
  str += "\t" + col_slot_context_.toString();
  return str;
}

std::vector<TargetInfo> target_exprs_to_infos(
    const std::vector<const hdk::ir::Expr*>& targets,
    const QueryMemoryDescriptor& query_mem_desc,
    bool bigint_count) {
  std::vector<TargetInfo> target_infos;
  for (const auto target_expr : targets) {
    auto target = get_target_info(target_expr, bigint_count);
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::NonGroupedAggregate) {
      set_notnull(target, false);
      target.type = target.type->withNullable(true);
    }
    target_infos.push_back(target);
  }
  return target_infos;
}

std::optional<size_t> QueryMemoryDescriptor::varlenOutputBufferElemSize() const {
  int64_t buffer_element_size{0};
  for (size_t i = 0; i < col_slot_context_.getSlotCount(); i++) {
    try {
      const auto slot_element_size = col_slot_context_.varlenOutputElementSize(i);
      if (slot_element_size < 0) {
        return std::nullopt;
      }
      buffer_element_size += slot_element_size;
    } catch (...) {
      continue;
    }
  }
  return buffer_element_size;
}

size_t QueryMemoryDescriptor::varlenOutputRowSizeToSlot(const size_t slot_idx) const {
  int64_t buffer_element_size{0};
  CHECK_LT(slot_idx, col_slot_context_.getSlotCount());
  for (size_t i = 0; i < slot_idx; i++) {
    try {
      const auto slot_element_size = col_slot_context_.varlenOutputElementSize(i);
      if (slot_element_size < 0) {
        continue;
      }
      buffer_element_size += slot_element_size;
    } catch (...) {
      continue;
    }
  }
  return buffer_element_size;
}

int8_t QueryMemoryDescriptor::warpSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return gpu_mgr->getSubGroupSize();
}

unsigned QueryMemoryDescriptor::gridSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return config_->exec.override_gpu_grid_size ? config_->exec.override_gpu_grid_size
                                              : gpu_mgr->getGridSize();
}

unsigned QueryMemoryDescriptor::blockSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return config_->exec.override_gpu_block_size ? config_->exec.override_gpu_block_size
                                               : gpu_mgr->getMaxBlockSize();
}

Data_Namespace::DataMgr* QueryMemoryDescriptor::getDataMgr() const {
  return data_mgr_;
}

BufferProvider* QueryMemoryDescriptor::getBufferProvider() const {
  CHECK(data_mgr_);
  return data_mgr_->getBufferProvider();
}
