/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "CompilationOptions.h"
#include "Shared/GpuPlatform.h"

#include <optional>

class CompilationOptionsBuilder {
 protected:
  CompilationOptions makeCompilationOptions(const ExecutorDeviceType dt);
  void setCodegenTraits(CompilationOptions& co, const std::optional<GpuMgrPlatform> p);
  void applyConfigSettings(CompilationOptions& co, const Config& cfg);

 public:
  virtual CompilationOptions build(const ExecutorDeviceType) = 0;
};

class CompilationOptionsDefaultBuilder : public CompilationOptionsBuilder {
 private:
  const Config& config_;
  const std::optional<GpuMgrPlatform> platform_;

 public:
  CompilationOptionsDefaultBuilder(const Config& cfg,
                                   const std::optional<GpuMgrPlatform> p = std::nullopt)
      : config_(cfg), platform_(p) {}
  CompilationOptions build(const ExecutorDeviceType dt) override;
};
