/**
 * Copyright 2022 OmniSci, Inc.
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include "AddressSpace.h"
namespace compiler {
enum class CallingConvDesc { C, SPIR_FUNC };
struct CodegenTraitsDescriptor {
  CodegenTraitsDescriptor(unsigned local_addr_space,
                          unsigned shared_addr_space,
                          unsigned global_addr_space,
                          CallingConvDesc calling_conv,
                          std::string_view triple)
      : local_addr_space_(local_addr_space)
      , shared_addr_space_(shared_addr_space)
      , global_addr_space_(global_addr_space)
      , conv_(calling_conv)
      , triple_(triple) {}
  CodegenTraitsDescriptor() {
    std::cout << "CodegenTraitsDescriptor shared_addr_space=" << shared_addr_space_
              << std::endl;
  };

  unsigned local_addr_space_{0};
  unsigned shared_addr_space_{0};
  unsigned global_addr_space_{0};
  CallingConvDesc conv_;
  std::string_view triple_{"DUMMY"};
};

const CodegenTraitsDescriptor cpu_cgen_traits_desc = {
    static_cast<unsigned>(CpuAddrSpace::kLocal),
    static_cast<unsigned>(CpuAddrSpace::kShared),
    static_cast<unsigned>(CpuAddrSpace::kGlobal),
    CallingConvDesc::C,
    std::string_view{""}};

const CodegenTraitsDescriptor cuda_cgen_traits_desc = {
    static_cast<unsigned>(CudaAddrSpace::kLocal),
    static_cast<unsigned>(CudaAddrSpace::kShared),
    static_cast<unsigned>(CudaAddrSpace::kGlobal),
    CallingConvDesc::C,
    std::string_view{"nvptx64-nvidia-cuda"}};

const CodegenTraitsDescriptor l0_cgen_traits_desc = {
    static_cast<unsigned>(L0AddrSpace::kLocal),
    static_cast<unsigned>(L0AddrSpace::kShared),
    static_cast<unsigned>(L0AddrSpace::kGlobal),
    CallingConvDesc::SPIR_FUNC,
    std::string_view{"spir64-unknown-unknown"}};
}  // namespace compiler
