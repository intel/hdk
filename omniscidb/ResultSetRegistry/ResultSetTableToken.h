/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSetTable.h"

#include "DataMgr/ChunkMetadata.h"
#include "DataProvider/TableFragmentsInfo.h"
#include "SchemaMgr/TableInfo.h"

#include "arrow/api.h"

namespace hdk {

class ResultSetRegistry;
class ResultSetTableToken;

using ResultSetTableTokenPtr = std::shared_ptr<const ResultSetTableToken>;

class ResultSetTableToken : public std::enable_shared_from_this<ResultSetTableToken> {
 public:
  class Iterator {
   public:
    using value_type = std::vector<TargetValue>;
    using difference_type = std::ptrdiff_t;
    using pointer = std::vector<TargetValue>*;
    using reference = std::vector<TargetValue>&;
    using iterator_category = std::input_iterator_tag;

    bool operator==(const Iterator& other) const { return rs_iter_ == other.rs_iter_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    value_type operator*() const {
      CHECK(rs_iter_.isValid());
      return *rs_iter_;
    }

    inline Iterator& operator++(void) {
      ++rs_iter_;
      maybeMoveToNextValid();
      return *this;
    }

    Iterator operator++(int) {
      Iterator iter(*this);
      ++(*this);
      return iter;
    }

   private:
    const ResultSetTableToken* table_;
    ResultSetRowIterator rs_iter_;
    size_t rs_idx_;
    bool translate_strings_;
    bool decimal_to_double_;

    Iterator(const ResultSetTableToken* table,
             bool translate_strings,
             bool decimal_to_double)
        : table_(table)
        , rs_iter_(table->resultSet(0)->rowIterator(translate_strings, decimal_to_double))
        , rs_idx_(0)
        , translate_strings_(translate_strings)
        , decimal_to_double_(decimal_to_double) {
      // In case the first ResultSet is empty.
      maybeMoveToNextValid();
    }

    void maybeMoveToNextValid() {
      while (!rs_iter_.isValid() && rs_idx_ != (table_->resultSetCount() - 1)) {
        ++rs_idx_;
        rs_iter_ = table_->resultSet(rs_idx_)->rowIterator(translate_strings_,
                                                           decimal_to_double_);
      }
    }

    friend class ResultSetTableToken;
  };

  ResultSetTableToken() = default;
  ResultSetTableToken(TableInfoPtr tinfo,
                      size_t row_count,
                      std::shared_ptr<ResultSetRegistry> registry);

  ResultSetTableToken(ResultSetTableToken&& other) = default;
  ResultSetTableToken& operator=(ResultSetTableToken&& other) = default;

  ResultSetTableToken(const ResultSetTableToken& other) = delete;
  ResultSetTableToken& operator=(const ResultSetTableToken& other) = delete;

  ~ResultSetTableToken();

  bool empty() const { return !registry_; }

  TableInfoPtr tableInfo() const { return tinfo_; }

  int dbId() const { return tinfo_->db_id; }
  int tableId() const { return tinfo_->table_id; }

  // Column ID <-> Index mapping. Use a significant offset to early catch
  // cases of indexes used as IDs and vice versa.
  static int columnId(size_t col_idx) { return static_cast<int>(col_idx + 2000); }
  static size_t columnIndex(int col_id) { return static_cast<size_t>(col_id - 2000); }

  size_t rowCount() const { return row_count_; }
  size_t colCount() const { return resultSet(0)->colCount(); }

  const hdk::ir::Type* colType(size_t col_idx) const {
    return resultSet(0)->colType(col_idx);
  }

  Iterator rowIterator(bool translate_strings, bool decimal_to_double) const {
    return Iterator(this, translate_strings, decimal_to_double);
  }

  size_t resultSetCount() const { return tinfo_->fragments; }
  ResultSetPtr resultSet(size_t idx) const;

  const std::string& tableName() const { return tinfo_->name; }

  void setTableStats(TableStats stats) const;

  ResultSetTableTokenPtr head(size_t n) const;
  ResultSetTableTokenPtr tail(size_t n) const;

  std::shared_ptr<arrow::Table> toArrow() const;

  std::vector<TargetValue> row(size_t row_idx,
                               bool translate_strings,
                               bool decimal_to_double) const;

  std::string toString() const {
    return "ResultSetTableToken(" + std::to_string(dbId()) + ":" +
           std::to_string(tableId()) + ")";
  }

  std::string description() const;
  std::string memoryDescription() const;
  std::string contentToString(bool header) const;

 private:
  void reset();

  TableInfoPtr tinfo_;
  size_t row_count_;
  std::shared_ptr<ResultSetRegistry> registry_;
};

}  // namespace hdk
