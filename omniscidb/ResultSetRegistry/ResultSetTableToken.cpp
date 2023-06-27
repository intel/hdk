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

std::vector<TargetValue> ResultSetTableToken::row(size_t row_idx,
                                                  bool translate_strings,
                                                  bool decimal_to_double) const {
  for (size_t rs_idx = 0; rs_idx < resultSetCount(); ++rs_idx) {
    auto rs = resultSet(rs_idx);
    if (rs->rowCount() > row_idx) {
      return rs->getRowAt(row_idx, translate_strings, decimal_to_double);
    }
    row_idx -= rs->rowCount();
  }
  throw std::runtime_error("Out-of-bound row index.");
}

std::string ResultSetTableToken::description() const {
  auto first_rs = resultSet(0);
  auto last_rs = resultSet(resultSetCount() - 1);
  size_t total_entries = first_rs->entryCount();
  for (size_t rs_idx = 1; rs_idx < resultSetCount(); ++rs_idx) {
    total_entries += resultSet(rs_idx)->entryCount();
  }
  std::ostringstream oss;
  oss << "Result Set Table Info" << std::endl;
  oss << "\tFragments: " << resultSetCount() << std::endl;
  oss << "\tLayout: " << first_rs->getQueryMemDesc().queryDescTypeToString() << std::endl;
  oss << "\tColumns: " << first_rs->colCount() << std::endl;
  oss << "\tRows: " << rowCount() << std::endl;
  oss << "\tEntry count: " << total_entries << std::endl;
  const std::string did_output_columnar =
      first_rs->didOutputColumnar() ? "True" : "False;";
  oss << "\tColumnar: " << did_output_columnar << std::endl;
  oss << "\tLazy-fetched columns: " << first_rs->getNumColumnsLazyFetched() << std::endl;
  const std::string is_direct_columnar_conversion_possible =
      first_rs->isDirectColumnarConversionPossible() ? "True" : "False";
  oss << "\tDirect columnar conversion possible: "
      << is_direct_columnar_conversion_possible << std::endl;

  size_t num_columns_zero_copy_columnarizable{0};
  for (size_t col_idx = 0; col_idx < first_rs->colCount(); col_idx++) {
    if (first_rs->isZeroCopyColumnarConversionPossible(col_idx)) {
      num_columns_zero_copy_columnarizable++;
    }
  }
  oss << "\tZero-copy columnar conversion columns: "
      << num_columns_zero_copy_columnarizable << std::endl;

  oss << "\tHas permutation: "
      << (first_rs->isPermutationBufferEmpty() ? "False" : "True") << std::endl;
  auto limit =
      last_rs->getLimit() ? row_count_ - last_rs->rowCount() + last_rs->getLimit() : 0;
  oss << "\tLimit: " << limit << std::endl;
  oss << "\tOffset: " << first_rs->getOffset() << std::endl;
  return oss.str();
}

std::string ResultSetTableToken::memoryDescription() const {
  return resultSet(0)->toString();
}

std::string ResultSetTableToken::contentToString(bool header) const {
  std::string res = resultSet(0)->contentToString(header);
  for (size_t rs_idx = 1; rs_idx < resultSetCount(); ++rs_idx) {
    res += resultSet(rs_idx)->contentToString(false);
  }
  return res;
}

}  // namespace hdk
