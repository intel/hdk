/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "InlineNullValues.h"
#include "SimpleAllocator.h"

#include "IR/OpTypeEnums.h"

#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace hdk::quantile {

namespace details {

class ChunkedArray {
 public:
  struct Chunk {
    int8_t* data;
    // Max number of elements that can be stored in a chunk. Element type is determined on
    // push when chunk is allocated. Elements of different types shouldn't be pushed into
    // the same chunk.
    size_t max_elems;
  };

  using ChunkVector = std::vector<Chunk>;

  // Random access iterator to be used with std::nth_element.
  template <typename T>
  class Iterator {
   public:
    typedef T value_type;
    typedef int64_t difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::random_access_iterator_tag iterator_category;

    Iterator(const ChunkVector* chunks, size_t chunk_idx, size_t chunk_offs)
        : chunks_(chunks), chunk_idx_(chunk_idx), chunk_offs_(chunk_offs) {}

    Iterator(const Iterator& other) = default;
    Iterator& operator=(const Iterator& other) = default;

    int64_t operator-(const Iterator& other) const {
      if (chunk_idx_ == other.chunk_idx_) {
        return chunk_offs_ - other.chunk_offs_;
      } else if (chunk_idx_ > other.chunk_idx_) {
        auto res =
            (*chunks_)[other.chunk_idx_].max_elems - other.chunk_offs_ + chunk_offs_;
        for (auto i = other.chunk_idx_ + 1; i < chunk_idx_; ++i) {
          res += (*chunks_)[i].max_elems;
        }
        return (int64_t)res;
      } else {
        auto res = (*chunks_)[chunk_idx_].max_elems - chunk_offs_ + other.chunk_offs_;
        for (auto i = chunk_idx_ + 1; i < other.chunk_idx_; ++i) {
          res += (*chunks_)[i].max_elems;
        }
        return -(int64_t)res;
      }
    }

    Iterator& operator+=(int64_t i) {
      if (i < 0) {
        operator-=(-i);
      } else {
        while (true) {
          int64_t rem = int64_t((*chunks_)[chunk_idx_].max_elems - chunk_offs_);
          if (rem > i) {
            chunk_offs_ += i;
            break;
          } else {
            ++chunk_idx_;
            chunk_offs_ = 0;
            i -= rem;
          }
        }
      }
      return *this;
    }

    Iterator operator+(int64_t i) const {
      Iterator res = *this;
      res += i;
      return res;
    }

    Iterator& operator-=(int64_t i) {
      if (i < 0) {
        operator+=(-i);
      } else {
        while ((int64_t)chunk_offs_ < i) {
          --chunk_idx_;
          i -= chunk_offs_;
          chunk_offs_ = (*chunks_)[chunk_idx_].max_elems;
        }
        chunk_offs_ -= i;
      }
      return *this;
    }

    Iterator operator-(int64_t i) const {
      Iterator res = *this;
      res -= i;
      return res;
    }

    Iterator& operator++() {
      ++chunk_offs_;
      if (chunk_offs_ == (*chunks_)[chunk_idx_].max_elems) {
        ++chunk_idx_;
        chunk_offs_ = 0;
      }
      return *this;
    }

    Iterator operator++(int) {
      Iterator res = *this;
      operator++();
      return res;
    }

    Iterator& operator--() {
      if (!chunk_offs_) {
        --chunk_idx_;
        chunk_offs_ = (*chunks_)[chunk_idx_].max_elems;
      }
      --chunk_offs_;
      return *this;
    }

    Iterator operator--(int) {
      Iterator res = *this;
      operator--();
      return res;
    }

    bool operator==(const Iterator& other) const {
      return chunk_idx_ == other.chunk_idx_ && chunk_offs_ == other.chunk_offs_;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

    bool operator<(const Iterator& other) const {
      if (chunk_idx_ != other.chunk_idx_) {
        return chunk_idx_ < other.chunk_idx_;
      }
      return chunk_offs_ < other.chunk_offs_;
    }

    bool operator<=(const Iterator& other) const {
      if (chunk_idx_ != other.chunk_idx_) {
        return chunk_idx_ < other.chunk_idx_;
      }
      return chunk_offs_ <= other.chunk_offs_;
    }

    T& operator*() const {
      auto data = reinterpret_cast<T*>((*chunks_)[chunk_idx_].data);
      return data[chunk_offs_];
    }

   private:
    const ChunkVector* chunks_;
    // Current chunk index. Can be equal to size of chunks_ vector for `end` iterator.
    size_t chunk_idx_;
    // Offset in the current chunk. Should always be less than chunk size when the
    // index points to a valid chunk.
    size_t chunk_offs_;
  };

  ChunkedArray(SimpleAllocator* allocator) : allocator_(allocator), cur_idx_(0) {}

  template <typename T>
  void push(T value) {
    // Check if we need to allocate a new chunk.
    if (chunks_.empty() || cur_idx_ == chunks_.back().max_elems) {
      if (thread_idx_ < 0) {
        thread_idx_ = tbb::this_task_arena::current_thread_index();
      } else if (thread_idx_ != tbb::this_task_arena::current_thread_index()) {
        // Pushing elements from different threads is not allowed bacause can cause
        // memory corruption.
        abort();
      }
      size_t size_to_allocate = std::max((size_t)64, (size_t)1 << chunks_.size()) << 10;
      Chunk chunk{allocator_->allocateSmallMtNoLock(size_to_allocate, thread_idx_),
                  size_to_allocate / sizeof(T)};
      chunks_.emplace_back(chunk);
      cur_idx_ = 0;
    }
    auto data = reinterpret_cast<T*>(chunks_.back().data);
    data[cur_idx_++] = value;
  }

  void merge(const ChunkedArray& other) {
    if (!other.chunks_.empty()) {
      // We are not going to add values to the current last chunk anymore, so fix-up
      // its size for proper iteration.
      if (!chunks_.empty()) {
        chunks_.back().max_elems = cur_idx_;
      }
      chunks_.insert(chunks_.end(), other.chunks_.begin(), other.chunks_.end());
      cur_idx_ = other.cur_idx_;
    }
  }

  template <typename T>
  Iterator<T> begin() {
    return Iterator<T>(&chunks_, 0, 0);
  }

  template <typename T>
  Iterator<T> end() {
    // Chunk offset in iterator is always expected to be less than chunk size. So, end
    // iterator for a full chunk is supposed to point the start of the next chunk.
    // Take care of it if all chunks are full.
    if (!chunks_.empty() && cur_idx_ == chunks_.back().max_elems) {
      return Iterator<T>(&chunks_, chunks_.size(), 0);
    }
    return Iterator<T>(&chunks_, chunks_.size() - 1, cur_idx_);
  }

  bool empty() const { return chunks_.empty(); }

  size_t size() const {
    if (chunks_.empty()) {
      return 0;
    }

    size_t res = cur_idx_;
    for (size_t i = 0; i < chunks_.size() - 1; ++i) {
      res += chunks_[i].max_elems;
    }
    return res;
  }

  void clear() {
    chunks_.clear();
    cur_idx_ = 0;
  }

 private:
  SimpleAllocator* allocator_;
  // All chunks except the last one should be full, i.e. they hold
  // chunk.max_elems elements.
  ChunkVector chunks_;
  // Insertion position in the last chunk.
  size_t cur_idx_;
  // Thread index working with this quantile object. We assume elements are pushed
  // by a single thread only. Index is determined on the first push. Merge can be
  // done from different threads.
  int thread_idx_ = -1;
};

class Quantile {
 public:
  Quantile(SimpleAllocator* simple_allocator)
      : values_(simple_allocator), finalized_(false) {}

  template <typename ValueType>
  void add(ValueType val) {
    // Shouldn't merge new data into finalized Quantile.
    if (finalized_) {
      abort();
    }
    values_.push<ValueType>(val);
  }

  // This method is used to compute current quantile value with specified quntile
  // parameter and interpolation. We don't support multiple quantile values computed
  // by a single object right now, so expect all calls to finalize and quantile methods
  // of the same object always uyse the same q and interpolation values.
  // finalize call with intermediate set to true doesn't actually finalizes the object
  // and allow following values collection and finalize calls. It can be used in various
  // debug scenarious to check the current object state, e.g. print intermediate ResultSet
  // before its reduction. The first finalize call with intermediate set to false means
  // we shouldn't add any more data or try to finalize it one more time.
  template <typename ValueType, typename ResultType>
  void finalize(double q, ir::Interpolation interpolation, bool intermediate = false) {
    if (finalized_) {
      abort();
    }

    if (values_.empty()) {
      finalized_ = !intermediate;
      static_assert(sizeof(ResultType) <= sizeof(res_));
      *reinterpret_cast<ResultType*>(&res_) = inline_null_value<ResultType>();
      return;
    }

    double pos = (values_.size() - 1) * q;
    size_t left_idx;
    size_t right_idx;
    switch (interpolation) {
      case ir::Interpolation::kLower:
        left_idx = right_idx = std::floor(pos);
        break;
      case ir::Interpolation::kHigher:
        left_idx = right_idx = std::ceil(pos);
        break;
      case ir::Interpolation::kNearest:
        left_idx = right_idx = std::round(pos);
        break;
      case ir::Interpolation::kMidpoint:
      case ir::Interpolation::kLinear:
        left_idx = std::floor(pos);
        right_idx = std::ceil(pos);
        break;
    }

    auto begin_iter = values_.begin<ValueType>();
    auto end_iter = values_.end<ValueType>();
    auto left_iter = begin_iter + left_idx;

    std::nth_element(begin_iter, left_iter, end_iter);
    auto left_value = *left_iter;
    ResultType res;
    if (left_idx != right_idx) {
      auto right_iter = left_iter + 1;
      std::nth_element(left_iter, right_iter, end_iter);
      // It is either midpoint or linear interpolation.
      double diff_coeff =
          interpolation == hdk::ir::Interpolation::kMidpoint ? 0.5 : pos - floor(pos);
      auto right_value = *right_iter;
      res = static_cast<ResultType>(left_value + (right_value - left_value) * diff_coeff);
    } else {
      res = static_cast<ResultType>(left_value);
    }

    static_assert(sizeof(ResultType) <= sizeof(res_));
    *reinterpret_cast<ResultType*>(&res_) = res;

    if (!intermediate) {
      finalized_ = true;
      values_.clear();
    }
  }

  // Parameters are designed for future updates to compute multiple quantiles
  // out of a single data set. It's not supported right now.
  // Currently, we don't expect this method to be called multiple times with
  // different parameters. We also don't expect finalize method to be called
  // with a different set of parameters than this method.
  template <typename ValueType, typename ResultType>
  ResultType quantile(double q, hdk::ir::Interpolation interpolation) {
    if (finalized_) {
      return *reinterpret_cast<const ResultType*>(&res_);
    }

    if (values_.empty()) {
      return inline_null_value<ResultType>();
    }

    // We are supposed to get hereonly for debug purposes when some intermediate
    // result is accessed. So we don't perform actual finalization and simply
    // compute the current quantile.
    finalize<ValueType, ResultType>(q, interpolation, true);
    static_assert(sizeof(ResultType) <= sizeof(res_));
    return *reinterpret_cast<const ResultType*>(&res_);
  }

  void merge(Quantile& other) {
    // Shouldn't merge new data into finalized Quantile.
    if (finalized_) {
      abort();
    }
    values_.merge(other.values_);
  }

  bool empty() const { return values_.empty(); }

 private:
  ChunkedArray values_;
  bool finalized_;
  int64_t res_;
};

}  // namespace details

using Quantile = details::Quantile;

}  // namespace hdk::quantile
