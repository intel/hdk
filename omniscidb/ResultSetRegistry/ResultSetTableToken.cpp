/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ResultSetTableToken.h"
#include "ResultSetRegistry.h"

namespace hdk {

ResultSetTableToken::ResultSetTableToken(TableInfoPtr tinfo,
                                         size_t row_count,
                                         std::shared_ptr<ResultSetRegistry> registry)
    : tinfo_(tinfo), row_count_(row_count), registry_(registry) {
  CHECK(registry);
  CHECK_EQ(getSchemaId(tinfo_->db_id), registry->getId());
}

ResultSetTableToken::~ResultSetTableToken() {
  reset();
}

ResultSetPtr ResultSetTableToken::resultSet(size_t idx) const {
  CHECK_LT(idx, resultSetCount());
  return registry_->get(*this, idx);
}

void ResultSetTableToken::reset() {
  if (!empty()) {
    registry_->drop(*this);
    registry_.reset();
    tinfo_.reset();
    row_count_ = 0;
  }
}

ResultSetTableTokenPtr ResultSetTableToken::head(size_t n) const {
  return registry_->head(*this, n);
}

ResultSetTableTokenPtr ResultSetTableToken::tail(size_t n) const {
  return registry_->tail(*this, n);
}

}  // namespace hdk
