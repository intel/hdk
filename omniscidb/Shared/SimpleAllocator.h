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

#pragma once

class SimpleAllocator {
 protected:
  ~SimpleAllocator() = default;

 public:
  virtual int8_t* allocate(const size_t num_bytes, const size_t thread_idx = 0) = 0;
  // This allocation method is supposed to be used by execution kernels for allocating
  // small memory batches. Callers are responsible for not using the same thread_idx
  // values from different threads. This enables lock-free thread local memory pools
  // usage for better performance. Implementations are likely to fallback to a regular
  // allocation for big memory chunks and for thread indexes exceeding cpu_count().
  virtual int8_t* allocateSmallMtNoLock(size_t size, size_t thread_idx = 0) {
    return allocate(size);
  }
};
