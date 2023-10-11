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

/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and
 * related data structures and types.
 */
#pragma once

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED 1

#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <mutex>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/BufferMgr/BufferSeg.h"
#include "Shared/boost_stacktrace.hpp"
#include "Shared/types.h"

class OutOfMemory : public std::runtime_error {
 public:
  OutOfMemory(size_t num_bytes)
      : std::runtime_error(parse_error_str("OutOfMemory", num_bytes)) {
    VLOG(1) << "Failed to allocate " << num_bytes << " bytes";
    VLOG(1) << boost::stacktrace::stacktrace();
  };

  OutOfMemory(const std::string& err) : std::runtime_error(parse_error_str(err, 0)) {
    VLOG(1) << "Failed with OutOfMemory, condition " << err;
    VLOG(1) << boost::stacktrace::stacktrace();
  };

  OutOfMemory(const std::string& err, size_t num_bytes)
      : std::runtime_error(parse_error_str(err, num_bytes)) {
    VLOG(1) << "Failed to allocate " << num_bytes << " bytes with condition " << err;
    VLOG(1) << boost::stacktrace::stacktrace();
  };

 private:
  std::string parse_error_str(const std::string& err, const size_t num_bytes = 0) {
    if (num_bytes) {
      return err + ": Failed to allocate " + std::to_string(num_bytes) + " bytes";
    } else {
      return "Failed to allocate memory with condition " + err;
    }
  }
};

class FailedToCreateFirstSlab : public OutOfMemory {
 public:
  FailedToCreateFirstSlab(size_t num_bytes)
      : OutOfMemory("FailedToCreateFirstSlab", num_bytes) {}
};

class FailedToCreateSlab : public OutOfMemory {
 public:
  FailedToCreateSlab(size_t num_bytes) : OutOfMemory("FailedToCreateSlab", num_bytes) {}
};

class TooBigForSlab : public OutOfMemory {
 public:
  TooBigForSlab(size_t num_bytes) : OutOfMemory("TooBigForSlab", num_bytes) {}
};

using namespace Data_Namespace;

namespace Buffer_Namespace {
struct Slab {
 private:
  static bool iterator_page_count_cmp_l(const SegmentList::iterator& lhs,
                                        const size_t num_pages) {
    return lhs->num_pages < num_pages;
  }
  struct SegmentListIterCmp {
    bool operator()(const SegmentList::iterator& lhs,
                    const SegmentList::iterator& rhs) const {
      if (lhs->num_pages < rhs->num_pages) {
        return true;
      }
      if (lhs->num_pages > rhs->num_pages) {
        return false;
      }
      return (lhs->start_page < rhs->start_page);
    }
  };

  std::set<SegmentList::iterator, SegmentListIterCmp> free_segment_index_;
  SegmentList segments_;

 public:
  Slab(const size_t page_count) : segments_(SegmentList{BufferSeg(0, page_count)}) {
    free_segment_index_.insert(segments_.begin());
  }

  SegmentList::iterator getFreeSegment(const size_t num_requested_pages);

  SegmentList::iterator insert(SegmentList::iterator& slab_buffer_pos,
                               BufferSeg& data_seg);

  SegmentList::iterator remove(SegmentList::iterator slab_buffer_pos);

  void index_insert(SegmentList::iterator& to_insert);
  void index_remove(SegmentList::iterator& at);
  void verify_index();

  SegmentList::iterator begin() { return segments_.begin(); }
  SegmentList::iterator end() { return segments_.end(); }

  const std::set<SegmentList::iterator, SegmentListIterCmp>& getFreeSegments() const {
    return free_segment_index_;
  };
  const SegmentList& getListToIndex() const { return segments_; };
};

struct MemoryData {
  size_t slabNum;
  int32_t startPage;
  size_t numPages;
  uint32_t touch;
  std::vector<int32_t> chunk_key;
  MemStatus memStatus;
};

struct MemoryInfo {
  size_t pageSize;
  size_t maxNumPages;
  size_t numPageAllocated;
  bool isAllocationCapped;
  std::vector<MemoryData> nodeMemoryData;
};

/**
 * @class   BufferMgr
 * @brief
 *
 * Note(s): Forbid Copying Idiom 4.1
 */

class BufferMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructs a BufferMgr object that allocates memSize bytes.
  BufferMgr(const int device_id,
            const size_t max_buffer_size,
            const size_t min_slab_size,
            const size_t max_slab_size,
            const size_t page_size,
            AbstractBufferMgr* parent_mgr = 0);

  /// Destructor
  ~BufferMgr() override;
  void reinit();

  std::string printSlab(size_t slab_num);
  std::string printSlabs() override;

  void clearSlabs();
  std::string printMap();
  void printSegs();
  std::string printSeg(SegmentList::iterator& seg_it);
  std::string keyToString(const ChunkKey& key);
  size_t getInUseSize() override;
  size_t getMaxSize() override;
  size_t getAllocated() override;
  size_t getMaxBufferSize();
  size_t getMaxSlabSize();
  size_t getPageSize();
  bool isAllocationCapped() override;

  /// Creates a chunk with the specified key and page size.
  AbstractBuffer* createBuffer(const ChunkKey& key,
                               const size_t page_size = 0,
                               const size_t initial_size = 0) override;

  AbstractBuffer* createZeroCopyBuffer(const ChunkKey& chunk_key,
                                       std::unique_ptr<AbstractDataToken> token) override;

  /// Deletes the chunk with the specified key
  void deleteBuffer(const ChunkKey& key, const bool purge = true) override;
  void deleteBuffersWithPrefix(const ChunkKey& key_prefix,
                               const bool purge = true) override;

  /// Returns the a pointer to the chunk with the specified key.
  AbstractBuffer* getBuffer(const ChunkKey& key, const size_t num_bytes = 0) override;

  std::unique_ptr<AbstractDataToken> getZeroCopyBufferMemory(const ChunkKey& key,
                                                             size_t numBytes) override;

  std::unique_ptr<AbstractDataToken> getZeroCopyColumnData(
      const ColumnRef& col_ref) override;

  /**
   * @brief Puts the contents of d into the Buffer with ChunkKey key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  bool isBufferOnDevice(const ChunkKey& key) override;
  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* dest_buffer,
                   const size_t num_bytes = 0) override;
  const DictDescriptor* getDictMetadata(int dict_id, bool load_dict = true) override {
    UNREACHABLE();
    return nullptr;
  }

  TableFragmentsInfo getTableMetadata(int db_id, int table_id) const override {
    UNREACHABLE();
    return TableFragmentsInfo{};
  }

  // Buffer API
  AbstractBuffer* alloc(const size_t num_bytes = 0) override;
  void free(AbstractBuffer* buffer) override;

  /// Returns the total number of bytes allocated.
  size_t size();
  size_t getNumChunks() override;

  SegmentList::iterator reserveBuffer(SegmentList::iterator& seg_it,
                                      const size_t num_bytes);
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata_vec,
                                       const ChunkKey& key_prefix) override;

  MemoryInfo getMemoryInfo();

 protected:
  const size_t
      max_buffer_pool_size_;    /// max number of bytes allocated for the buffer pool
  const size_t min_slab_size_;  /// minimum size of the individual memory allocations that
                                /// compose the buffer pool (up to maxBufferSize_)
  const size_t max_slab_size_;  /// size of the individual memory allocations that compose
                                /// the buffer pool (up to maxBufferSize_)
  const size_t page_size_;
  std::vector<int8_t*> slabs_;  /// vector of beginning memory addresses for each
                                /// allocation of the buffer pool
  std::vector<Slab> slab_segments_;

 private:
  BufferMgr(const BufferMgr&);             // private copy constructor
  BufferMgr& operator=(const BufferMgr&);  // private assignment
  void removeSegment(SegmentList::iterator& seg_it);
  SegmentList::iterator findFreeSegmentInSlab(const size_t slab_num,
                                              const size_t num_pages_requested);
  int getBufferId();
  virtual void addSlab(const size_t slab_size) = 0;
  virtual void freeAllMem() = 0;
  virtual void allocateBuffer(SegmentList::iterator seg_it,
                              const size_t page_size,
                              const size_t num_bytes) = 0;
  virtual AbstractBuffer* allocateZeroCopyBuffer(
      const size_t page_size,
      std::unique_ptr<AbstractDataToken> token) {
    UNREACHABLE();
    return nullptr;
  }
  void clear();

  std::mutex chunk_index_mutex_;
  std::mutex sized_segs_mutex_;
  std::mutex unsized_segs_mutex_;
  std::mutex buffer_id_mutex_;
  // This map is used for granular locks in case of parallel requests
  // to fetch the same chunk that is not in chunk_index yet. Access
  // to this map should be synced throug chunk_index_mutex_.
  std::map<ChunkKey, std::shared_ptr<std::condition_variable>> in_progress_buffer_cvs_;

  std::map<ChunkKey, SegmentList::iterator> chunk_index_;
  size_t max_buffer_pool_num_pages_;  // max number of pages for buffer pool
  size_t num_pages_allocated_;
  size_t min_num_pages_per_slab_;
  size_t max_num_pages_per_slab_;
  size_t current_max_slab_page_size_;
  bool allocations_capped_;
  AbstractBufferMgr* parent_mgr_;
  int max_buffer_id_;
  unsigned int buffer_epoch_;

  SegmentList unsized_segs_;

  SegmentList::iterator evict(SegmentList::iterator& evict_start,
                              const size_t num_pages_requested,
                              const int slab_num);

  /**
   * @brief Gets a buffer of required size and returns an iterator to it
   *
   * If possible, this function will just select a free buffer of
   * sufficient size and use that. If not, it will evict as many
   * non-pinned but used buffers as needed to have enough space for the
   * buffer
   *
   * @return An iterator to the reserved buffer. We guarantee that this
   * buffer won't be evicted by PINNING it - caller should change this to
   * USED if applicable
   *
   */
  SegmentList::iterator findFreeSegment(size_t num_bytes);
};

}  // namespace Buffer_Namespace
