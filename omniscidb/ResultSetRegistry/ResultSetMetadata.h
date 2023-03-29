/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/ChunkMetadata.h"
#include "ResultSet/ResultSet.h"

namespace hdk {

ChunkMetadataMap synthesizeMetadata(const ResultSet* rows);

}
