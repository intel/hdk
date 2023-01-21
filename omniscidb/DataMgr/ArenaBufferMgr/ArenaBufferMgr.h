/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/functional/hash.hpp>
#include <unordered_map>

#include "DataMgr/ArenaBufferMgr/ArenaBuffer.h"
#include "DataMgr/BufferMgr/Buffer.h"

#include "DataMgr/Allocators/ArenaAllocator.h"

namespace Data_Namespace {

struct RowArena {
  RowArena(const size_t num_cols) { chunk_buffers.resize(num_cols); }

  std::vector<std::unique_ptr<ArenaBuffer>> chunk_buffers;

  ArenaBuffer* getChunkBuffer(const int col_id) {
    CHECK_LT(size_t(col_id), chunk_buffers.size());
    return chunk_buffers[col_id].get();
  }

  ArenaBuffer* insertChunkBuffer(const int col_id,
                                 std::unique_ptr<ArenaBuffer>&& buffer) {
    CHECK_LT(size_t(col_id), chunk_buffers.size());
    CHECK(!chunk_buffers[col_id]);
    chunk_buffers[col_id] = std::move(buffer);
    return chunk_buffers[col_id].get();
  }

  Arena arena;
};

struct TableArena {
  TableArena(const size_t num_cols) : num_cols(num_cols) {}

  // fragment_id
  std::unordered_map<int, std::unique_ptr<RowArena>> row_arenas;

  RowArena* getArenaForFragment(const int fragment_id) {
    if (row_arenas.find(fragment_id) == row_arenas.end()) {
      // initialize row arena
      auto row_arena = std::make_unique<RowArena>(num_cols);
      row_arenas.insert(
          std::pair<int, std::unique_ptr<RowArena>>(fragment_id, std::move(row_arena)));
    }
    auto row_arena_itr = row_arenas.find(fragment_id);
    CHECK(row_arena_itr != row_arenas.end());
    return row_arena_itr->second.get();
  }

  // number of columns in this table, for initializing row arenas
  const size_t num_cols;
};

struct ArenaKey {
  ChunkKey key;

  bool operator==(const ArenaKey& b) const { return key == b.key; }
};

struct ArenaKeyHash {
  size_t operator()(const ArenaKey& val) const {
    if (val.key.size() == 0) {
      return 0;
    }
    size_t h = 0;

    boost::hash_combine(h, val.key.front());
    for (size_t i = 1; i < val.key.size(); i++) {
      boost::hash_combine(h, val.key[i]);
    }
    return h;
  }
};

class ArenaBufferMgr {
 public:
  ArenaBufferMgr(AbstractBufferMgr* storage_mgr) {
    CHECK(storage_mgr);
    storage_mgr_ = dynamic_cast<PersistentStorageMgr*>(storage_mgr);
    CHECK(storage_mgr_);
  }

  AbstractBuffer* getChunkBuffer(const ChunkKey& key, const size_t num_bytes) {
    std::lock_guard<std::mutex> big_lock(global_mutex_);  // TODO: remove

    ArenaKey a_key{key};

    const auto table_key =
        std::make_pair(key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX]);
    if (arenas_per_table_.find(table_key) == arenas_per_table_.end()) {
      initializeTableArena(table_key);
    }
    auto table_arena_itr = arenas_per_table_.find(table_key);
    CHECK(table_arena_itr != arenas_per_table_.end());
    auto table_arena = table_arena_itr->second.get();
    CHECK(table_arena);

    auto row_arena = table_arena->getArenaForFragment(key[CHUNK_KEY_FRAGMENT_IDX]);
    CHECK(row_arena);

    // this is gonna be nullptr, need to populate it
    auto chunk_buffer = row_arena->getChunkBuffer(key[CHUNK_KEY_COLUMN_IDX]);
    if (chunk_buffer == nullptr) {
      chunk_buffer = fetchChunkBufferFromStorage(key, row_arena, num_bytes);
    }
    return chunk_buffer;
  }

 private:
  void initializeTableArena(const std::pair<int, int>& table_key) {
    auto table_metadata =
        storage_mgr_->getTableMetadata(table_key.first, table_key.second);
    CHECK_GE(table_metadata.fragments.size(), size_t(1));
    auto& first_fragment = table_metadata.fragments.front();
    const auto& chunk_metadata = first_fragment.getChunkMetadataMapPhysical();
    CHECK_GT(chunk_metadata.size(), size_t(0));
    size_t num_cols = chunk_metadata.size();

    auto table_arena = std::make_unique<TableArena>(num_cols);
    arenas_per_table_.insert(std::make_pair(table_key, std::move(table_arena)));
  }

  ArenaBuffer* fetchChunkBufferFromStorage(const ChunkKey& key,
                                           RowArena* row_arena,
                                           const size_t num_bytes) {
    CHECK_GT(num_bytes, size_t(0));  // TODO: needed?

    auto chunk_ptr = reinterpret_cast<int8_t*>(row_arena->arena.allocate(
        num_bytes));  // we should try and keep track of the block,
                      // but just allocate the chunk for now
    auto arena_buffer = std::make_unique<ArenaBuffer>(
        chunk_ptr, num_bytes);  // todo: set chunk_ptr addr here
    storage_mgr_->fetchBuffer(key, arena_buffer.get(), num_bytes);

    return row_arena->insertChunkBuffer(key[CHUNK_KEY_COLUMN_IDX],
                                        std::move(arena_buffer));
  }

  // index outer container by db and table (key0, key1)
  // then index by fragment (key3)
  // then by column (key2)
  // sigh

  // db_id, table
  std::unordered_map<std::pair<int, int>, std::unique_ptr<TableArena>> arenas_per_table_;

  PersistentStorageMgr* storage_mgr_;

  std::mutex global_mutex_;
};

}  // namespace Data_Namespace