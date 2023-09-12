/*
 * Copyright 2020 OmniSci, Inc.
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
 * @file    QueryFragmentDescriptor.h
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Descriptor for the fragments required for an execution kernel.
 */

#pragma once

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "DataMgr/ChunkMetadata.h"
#include "Logger/Logger.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/CostModel/Dispatchers/ExecutionPolicy.h"

namespace Buffer_Namespace {
struct MemoryInfo;
}

class Executor;
class InputDescriptor;
struct InputTableInfo;
struct RelAlgExecutionUnit;

struct FragmentsPerTable {
  int db_id;
  int table_id;
  std::vector<size_t> fragment_ids;
};

using FragmentsList = std::vector<FragmentsPerTable>;
using TableFragments = std::vector<FragmentInfo>;

struct ExecutionKernelDescriptor {
  int device_id;
  FragmentsList fragments;
  std::optional<size_t> outer_tuple_count;  // only for fragments with an exact tuple
                                            // count available in metadata
};

class QueryFragmentDescriptor {
 public:
  QueryFragmentDescriptor(const RelAlgExecutionUnit& ra_exe_unit,
                          const std::vector<InputTableInfo>& query_infos,
                          const std::vector<Buffer_Namespace::MemoryInfo>& gpu_mem_infos,
                          const double gpu_input_mem_limit_percent,
                          const std::vector<size_t> allowed_outer_fragment_indices);

  static void computeAllTablesFragments(
      std::map<TableRef, const TableFragments*>& all_tables_fragments,
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& query_infos);

  void buildFragmentKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                              const std::vector<uint64_t>& frag_offsets,
                              const policy::ExecutionPolicy* policy,
                              Executor* executor,
                              compiler::CodegenTraitsDescriptor cgen_traits_desc);

  /**
   * Dispatch according to policy
   */
  template <typename DISPATCH_FCN>
  void dispatchKernelsToDevices(DISPATCH_FCN dispatcher_f,
                                const RelAlgExecutionUnit& ra_exe_unit,
                                const policy::ExecutionPolicy* policy) const {
    std::unordered_map<ExecutorDeviceType, std::unordered_map<int, size_t>>
        execution_kernel_index;
    size_t tuple_count = 0;
    for (const auto& device_type_itr : execution_kernels_per_device_) {
      if (policy->getExecutionMode(device_type_itr.first) ==
          ExecutorDispatchMode::KernelPerFragment) {
        for (const auto& device_itr : device_type_itr.second) {
          CHECK(execution_kernel_index[device_type_itr.first]
                    .insert(std::make_pair(device_itr.first, size_t(0)))
                    .second);
        }
      }
    }

    for (const auto& device_type_itr : execution_kernels_per_device_) {
      if (policy->getExecutionMode(device_type_itr.first) ==
          ExecutorDispatchMode::MultifragmentKernel) {
        for (const auto& device_itr : device_type_itr.second) {
          const auto& execution_kernels = device_itr.second;
          const auto& fragments_list = execution_kernels.front().fragments;
          dispatcher_f(
              device_itr.first, fragments_list, rowid_lookup_key_, device_type_itr.first);
        }
      } else {
        bool dispatch_finished = false;
        while (!dispatch_finished) {
          dispatch_finished = true;
          for (const auto& device_itr : device_type_itr.second) {
            auto& kernel_idx =
                execution_kernel_index[device_type_itr.first][device_itr.first];
            if (kernel_idx < device_itr.second.size()) {
              dispatch_finished = false;
              const auto& execution_kernel = device_itr.second[kernel_idx++];
              dispatcher_f(device_itr.first,
                           execution_kernel.fragments,
                           rowid_lookup_key_,
                           device_type_itr.first);
              if (terminateDispatchMaybe(tuple_count, ra_exe_unit, execution_kernel)) {
                return;
              }
            }
          }
        }
      }
    }
  }

  bool shouldCheckWorkUnitWatchdog() const {
    return rowid_lookup_key_ < 0 && !execution_kernels_per_device_.empty();
  }

 protected:
  std::vector<size_t> allowed_outer_fragment_indices_;
  size_t outer_fragments_size_ = 0;
  int64_t rowid_lookup_key_ = -1;

  std::map<TableRef, const TableFragments*> selected_tables_fragments_;

  std::map<ExecutorDeviceType, std::map<int, std::vector<ExecutionKernelDescriptor>>>
      execution_kernels_per_device_;

  double gpu_input_mem_limit_percent_;
  std::map<size_t, size_t> tuple_count_per_gpu_device_;
  std::map<size_t, size_t> available_gpu_mem_bytes_;

  void buildFragmentPerKernelMapForUnion(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<uint64_t>& frag_offsets,
      const policy::ExecutionPolicy* policy,
      const size_t num_bytes_for_row,
      Executor* executor,
      compiler::CodegenTraitsDescriptor cgen_traits_desc);

  void buildFragmentPerKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<uint64_t>& frag_offsets,
                                 const policy::ExecutionPolicy* policy,
                                 const size_t num_bytes_for_row,
                                 Executor* executor,
                                 compiler::CodegenTraitsDescriptor cgen_traits_desc);

  void buildMultifragKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<uint64_t>& frag_offsets,
                               const policy::ExecutionPolicy* policy,
                               const size_t num_bytes_for_row,
                               Executor* executor,
                               compiler::CodegenTraitsDescriptor cgen_traits_desc);

  void buildFragmentPerKernelForTable(const TableFragments* fragments,
                                      const RelAlgExecutionUnit& ra_exe_unit,
                                      const InputDescriptor& table_desc,
                                      const std::vector<uint64_t>& frag_offsets,
                                      const policy::ExecutionPolicy* policy,
                                      const size_t num_bytes_for_row,
                                      const std::optional<size_t> table_desc_offset,
                                      Executor* executor,
                                      compiler::CodegenTraitsDescriptor cgen_traits_desc);

  bool terminateDispatchMaybe(size_t& tuple_count,
                              const RelAlgExecutionUnit& ra_exe_unit,
                              const ExecutionKernelDescriptor& kernel) const;

  void checkDeviceMemoryUsage(const FragmentInfo& fragment,
                              const int device_id,
                              const size_t num_cols);
};

std::ostream& operator<<(std::ostream&, FragmentsPerTable const&);
