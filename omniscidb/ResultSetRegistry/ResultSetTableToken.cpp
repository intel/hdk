/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ResultSetTableToken.h"
#include "ResultSetRegistry.h"

#include "ResultSet/ArrowResultSet.h"
#include "Shared/ArrowUtil.h"

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

std::shared_ptr<arrow::Table> ResultSetTableToken::toArrow() const {
  auto first_rs = resultSet(0);
  std::vector<std::string> col_names;
  for (size_t col_idx = 0; col_idx < first_rs->colCount(); ++col_idx) {
    col_names.push_back(first_rs->colName(col_idx));
  }

  std::vector<std::shared_ptr<arrow::Table>> converted_tables;
  for (size_t rs_idx = 0; rs_idx < resultSetCount(); ++rs_idx) {
    ArrowResultSetConverter converter(resultSet(rs_idx), col_names, -1);
    converted_tables.push_back(converter.convertToArrowTable());
  }

  if (converted_tables.size() == (size_t)1) {
    return converted_tables.front();
  }

  ARROW_ASSIGN_OR_THROW(auto res, arrow::ConcatenateTables(converted_tables));
  return res;
}

}  // namespace hdk
