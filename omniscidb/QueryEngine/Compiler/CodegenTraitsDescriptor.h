/**
 * Copyright 2022 OmniSci, Inc.
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
namespace compiler {
enum class CallingConvDesc { C, SPIR_FUNC };
struct CodegenTraitsDescriptor {
  CodegenTraitsDescriptor(unsigned local_addr_space,
                          unsigned global_addr_space,
                          CallingConvDesc calling_conv,
                          std::string_view triple)
      : local_addr_space_(local_addr_space)
      , global_addr_space_(global_addr_space)
      , conv_(calling_conv)
      , triple_(triple) {}
  CodegenTraitsDescriptor(){};

  unsigned local_addr_space_{0};
  unsigned global_addr_space_{0};
  CallingConvDesc conv_;
  std::string_view triple_{"DUMMY"};
};

const CodegenTraitsDescriptor cpu_cgen_traits_desc = {0,
                                                      0,
                                                      CallingConvDesc::C,
                                                      std::string_view{""}};

const CodegenTraitsDescriptor cuda_cgen_traits_desc = {
    4,
    1,
    CallingConvDesc::C,
    std::string_view{"nvptx64-nvidia-cuda"}};

const CodegenTraitsDescriptor l0_cgen_traits_desc = {
    4,
    1,
    CallingConvDesc::SPIR_FUNC,
    std::string_view{"spir64-unknown-unknown"}};
}  // namespace compiler
