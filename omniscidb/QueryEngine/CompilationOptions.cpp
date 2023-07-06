/*
 * Copyright 2022 OmniSci, Inc.
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

#include "QueryEngine/CompilationOptions.h"
#include <ostream>

#ifndef __CUDACC__
std::ostream& operator<<(std::ostream& os, const ExecutionOptions& eo) {
  os << "output_columnar_hint=" << eo.output_columnar_hint << "\n"
     << "allow_multifrag=" << eo.allow_multifrag << "\n"
     << "just_explain=" << eo.just_explain << "\n"
     << "allow_loop_joins=" << eo.allow_loop_joins << "\n"
     << "with_watchdog=" << eo.with_watchdog << "\n"
     << "jit_debug=" << eo.jit_debug << "\n"
     << "just_validate=" << eo.just_validate << "\n"
     << "with_dynamic_watchdog=" << eo.with_dynamic_watchdog << "\n"
     << "find_push_down_candidates=" << eo.find_push_down_candidates << "\n"
     << "just_calcite_explain=" << eo.just_calcite_explain << "\n"
     << "gpu_input_mem_limit_percent=" << eo.gpu_input_mem_limit_percent << "\n"
     << "allow_runtime_query_interrupt=" << eo.allow_runtime_query_interrupt << "\n"
     << "running_query_interrupt_freq=" << eo.running_query_interrupt_freq << "\n"
     << "pending_query_interrupt_freq=" << eo.pending_query_interrupt_freq << "\n"
     << "multifrag_result=" << eo.multifrag_result << "\n"
     << "preserve_order=" << eo.preserve_order << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ExecutorOptLevel& ol) {
  switch (ol) {
    case ExecutorOptLevel::Default:
      return os << "ExecutorOptLevel::Default";

    case ExecutorOptLevel::ReductionJIT:
      return os << "ExecutorOptLevel::ReductionJIT";

    default:
      return os << "Unknown ExecutorOptLevel";
  }
}
#endif
