/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CompilationOptionsBuilder.h"
#include "Logger/Logger.h"

CompilationOptions CompilationOptionsBuilder::makeCompilationOptions(
    const ExecutorDeviceType dt) {
  return CompilationOptions{dt,
                            /*hoist_literals=*/true,
                            /*opt_level=*/ExecutorOptLevel::Default,
                            /*with_dynamic_watchdog=*/false,
                            /*allow_lazy_fetch=*/true,
                            /*filter_on_delted_column=*/true,
                            /*explain_type=*/ExecutorExplainType::Default,
                            /*register_intel_jit_listener=*/false,
                            /*use_groupby_buffer_desc=*/false,
                            /*codegen_traits_desc=*/compiler::cpu_cgen_traits_desc};
}

void CompilationOptionsBuilder::setCodegenTraits(
    CompilationOptions& co,
    const std::optional<GpuMgrPlatform> platform) {
  CHECK_EQ(co.device_type, ExecutorDeviceType::GPU);
  co.codegen_traits_desc = CompilationOptions::getCgenTraitsDesc(
      co.device_type,
      (platform.has_value() ? platform.value() == GpuMgrPlatform::L0 : false));
}

void CompilationOptionsBuilder::applyConfigSettings(CompilationOptions& co,
                                                    const Config& cfg) {
  co.hoist_literals = cfg.exec.codegen.hoist_literals;
  co.device_type = cfg.exec.cpu_only ? ExecutorDeviceType::CPU : co.device_type;
}

CompilationOptions CompilationOptionsDefaultBuilder::build(const ExecutorDeviceType dt) {
  CHECK(config_);
  auto co = makeCompilationOptions(dt);
  applyConfigSettings(co, *config_);
  setCodegenTraits(co, *platform_);
  return co;
}
