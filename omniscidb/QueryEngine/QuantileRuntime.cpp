/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/quantile.h"

#define DEF_AGG_QUANTILE_IMPL(val_type, suffix)                                    \
  extern "C" RUNTIME_EXPORT DEVICE void agg_quantile_impl_##suffix(int64_t* agg,   \
                                                                   val_type val) { \
    auto* quantile = reinterpret_cast<hdk::quantile::Quantile*>(*agg);             \
    quantile->add<val_type>(val);                                                  \
  }

DEF_AGG_QUANTILE_IMPL(int8_t, int8)
DEF_AGG_QUANTILE_IMPL(int16_t, int16)
DEF_AGG_QUANTILE_IMPL(int32_t, int32)
DEF_AGG_QUANTILE_IMPL(int64_t, int64)
DEF_AGG_QUANTILE_IMPL(float, float)
DEF_AGG_QUANTILE_IMPL(double, double)
