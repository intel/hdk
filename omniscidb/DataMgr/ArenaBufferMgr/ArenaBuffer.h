/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/AbstractBuffer.h"

namespace Data_Namespace {

class ArenaBuffer : public AbstractBuffer {
 public:
  ArenaBuffer(int8_t* buffer_ptr, const size_t num_bytes)
      : AbstractBuffer(/*device_id=*/-1), ptr_(buffer_ptr), num_bytes_(num_bytes) {}

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int dst_device_id = -1) {
    CHECK(false);  // TODO
  }
  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int src_device_id = -1) {
    CHECK(false);
  }
  void reserve(size_t num_bytes) override { CHECK_EQ(num_bytes, num_bytes_); }
  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int device_id = -1) override {
    CHECK(false);
  }
  int8_t* getMemoryPtr() override { return ptr_; }
  size_t pageCount() const override { return 0; }
  size_t pageSize() const override { return 0; }
  size_t reservedSize() const override { return num_bytes_; }

  MemoryLevel getType() const override { return MemoryLevel::CPU_LEVEL; }

 private:
  int8_t* ptr_;
  const size_t num_bytes_;
};

}  // namespace Data_Namespace