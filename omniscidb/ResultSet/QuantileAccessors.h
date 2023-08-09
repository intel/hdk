/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "TargetValue.h"

#include "Shared/TargetInfo.h"
#include "Shared/quantile.h"

void finalizeQuantile(hdk::quantile::Quantile* quantile, const TargetInfo& target_info);
TargetValue getQuantile(hdk::quantile::Quantile* quantile, const TargetInfo& target_info);
InternalTargetValue getQuantileInternal(hdk::quantile::Quantile* quantile,
                                        const TargetInfo& target_info);
