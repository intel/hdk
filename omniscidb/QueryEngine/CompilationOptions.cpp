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

std::ostream& operator<<(std::ostream& os, const ExecutorOptLevel& eol) {
  switch (eol) {
    case ExecutorOptLevel::Default:
      return os << "ExecutorOptLevel::Default";
    case ExecutorOptLevel::ReductionJIT:
      return os << "ExecutorOptLevel::ReductionJIT";
    default:
      return os << "ExecutorOptLevel::UNKNOWN";
  }
}

std::ostream& operator<<(std::ostream& os, const ExecutorExplainType& eet) {
  switch (eet) {
    case ExecutorExplainType::Default:
      return os << "ExecutorExplainType::Default";
    case ExecutorExplainType::Optimized:
      return os << "ExecutorExplainType::Optimized";
    default:
      return os << "ExecutorExplainType::UNKNOWN";
  }
}

std::ostream& operator<<(std::ostream& os, const compiler::CallingConvDesc& desc) {
  switch (desc) {
    case compiler::CallingConvDesc::C:
      return os << "CallingConvDesc::C";
    case compiler::CallingConvDesc::SPIR_FUNC:
      return os << "CallingConvDesc::SPIR_FUNC";
    default:
      return os << "CallingConvDesc::UNKNOWN";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const compiler::CodegenTraitsDescriptor& desc) {
  os << "{local=" << desc.local_addr_space_ << ",global=" << desc.global_addr_space_
     << ",shared=" << desc.smem_addr_space_ << ",conv=" << desc.conv_
     << ",trpile=" << desc.triple_ << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const CompilationOptions& co) {
  os << "device_type=" << co.device_type << "\n"
     << "hoist_literals=" << co.hoist_literals << "\n"
     << "opt_level=" << co.opt_level << "\n"
     << "with_dynamic_watchdog=" << co.with_dynamic_watchdog << "\n"
     << "allow_lazy_fetch=" << co.allow_lazy_fetch << "\n"
     << "filter_on_deleted_column=" << co.filter_on_deleted_column << "\n"
     << "explain_type=" << co.explain_type << "\n"
     << "register_intel_jit_listener=" << co.register_intel_jit_listener << "\n"
     << "use_groupby_buffer_desc=" << co.use_groupby_buffer_desc << "\n"
     << "codegen_traits_desc=" << co.codegen_traits_desc << "\n";
  return os;
}
#endif
