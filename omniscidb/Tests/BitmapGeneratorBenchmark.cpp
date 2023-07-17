/**
 * Copyright 2023 Intel Corporation
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

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>

#include "Logger/Logger.h"
#include "ResultSet/BitmapGenerators.h"

size_t g_rand_seed = 42;

// percent_nulls is the percentage of time the distribution returns true, expressed in
// decimal form (e.g. / 100)
template <typename T>
static std::unique_ptr<T[]> gen_data(size_t size, const T null_val, float percent_nulls) {
  auto data_buf = std::make_unique<T[]>(size);
  auto data_buf_ptr = data_buf.get();

  std::mt19937 rand_gen(g_rand_seed);
  std::bernoulli_distribution dist(percent_nulls);
  for (size_t i = 0; i < size; i++) {
    if (dist(rand_gen)) {
      data_buf_ptr[i] = null_val;
    } else {
      data_buf_ptr[i] = i % std::numeric_limits<T>::max();
    }
  }

  return data_buf;
}

static auto alloc_bitmap(size_t input_sz) {
  // allocate extra space to ensure we can align
  size_t crt_sz = input_sz;
  constexpr size_t alignment = 64;
  if (crt_sz % alignment != 0) {
    crt_sz += (alignment - (crt_sz % alignment));
  }

  void* bitmap_buf_ptr = std::aligned_alloc(alignment, crt_sz);
  auto bitmap_buf_owned = std::unique_ptr<char[], void (*)(char*)>(
      reinterpret_cast<char*>(bitmap_buf_ptr), [](char* ptr) { std::free(ptr); });
  return bitmap_buf_owned;
}

static void null_bitmap_8(benchmark::State& state) {
  // generate a 4096 element buffer with 25% nulls (+ a few more since the null sentinel
  // will match the max size in the buffer)
  const size_t num_elems = 4096;
  auto data_buf = gen_data<uint8_t>(num_elems, std::numeric_limits<uint8_t>::max(), 0.25);
  auto bitmap_ptr_owned = alloc_bitmap(ceil(num_elems / 8.));

  auto initial_null_count =
      gen_null_bitmap_8(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                        data_buf.get(),
                        num_elems,
                        std::numeric_limits<uint8_t>::max());

  for (auto _ : state) {
    auto null_count =
        gen_null_bitmap_8(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                          data_buf.get(),
                          num_elems,
                          /*null_value=*/std::numeric_limits<uint8_t>::max());
    CHECK_EQ(null_count, initial_null_count);  // ensure null counts match
  }
}

static void null_bitmap_16(benchmark::State& state) {
  // generate a 4096 element buffer with 25% nulls (+ a few more since the null sentinel
  // will match the max size in the buffer)
  const size_t num_elems = 4096;
  auto data_buf =
      gen_data<uint16_t>(num_elems, std::numeric_limits<uint16_t>::max(), 0.25);
  auto bitmap_ptr_owned = alloc_bitmap(ceil(num_elems / 8.));

  auto initial_null_count =
      gen_null_bitmap_16(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                         data_buf.get(),
                         num_elems,
                         /*null_value=*/std::numeric_limits<uint16_t>::max());

  for (auto _ : state) {
    auto null_count =
        gen_null_bitmap_16(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                           data_buf.get(),
                           num_elems,
                           /*null_value=*/std::numeric_limits<uint16_t>::max());
    CHECK_EQ(null_count, initial_null_count);  // ensure null counts match
  }
}

static void null_bitmap_32(benchmark::State& state) {
  // generate a 4096 element buffer with 25% nulls (+ a few more since the null sentinel
  // will match the max size in the buffer)
  const size_t num_elems = 4096;
  auto data_buf =
      gen_data<uint32_t>(num_elems, std::numeric_limits<uint32_t>::max(), 0.25);
  auto bitmap_ptr_owned = alloc_bitmap(ceil(num_elems / 8.));

  auto initial_null_count =
      gen_null_bitmap_32(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                         data_buf.get(),
                         num_elems,
                         /*null_value=*/std::numeric_limits<uint32_t>::max());

  for (auto _ : state) {
    auto null_count =
        gen_null_bitmap_32(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                           data_buf.get(),
                           num_elems,
                           /*null_value=*/std::numeric_limits<uint32_t>::max());
    CHECK_EQ(null_count, initial_null_count);  // ensure null counts match
  }
}

static void null_bitmap_64(benchmark::State& state) {
  // generate a 4096 element buffer with 25% nulls (+ a few more since the null sentinel
  // will match the max size in the buffer)
  const size_t num_elems = 4096;
  auto data_buf =
      gen_data<uint64_t>(num_elems, std::numeric_limits<uint64_t>::max(), 0.25);
  auto bitmap_ptr_owned = alloc_bitmap(ceil(num_elems / 8.));

  auto initial_null_count =
      gen_null_bitmap_64(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                         data_buf.get(),
                         num_elems,
                         /*null_value=*/std::numeric_limits<uint64_t>::max());

  for (auto _ : state) {
    auto null_count =
        gen_null_bitmap_64(reinterpret_cast<uint8_t*>(bitmap_ptr_owned.get()),
                           data_buf.get(),
                           num_elems,
                           /*null_value=*/std::numeric_limits<uint64_t>::max());
    CHECK_EQ(null_count, initial_null_count);  // ensure null counts match
  }
}

BENCHMARK(null_bitmap_8);
BENCHMARK(null_bitmap_16);
BENCHMARK(null_bitmap_32);
BENCHMARK(null_bitmap_64);

BENCHMARK_MAIN();
