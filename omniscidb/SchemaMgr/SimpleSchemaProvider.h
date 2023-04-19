/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "IR/Context.h"
#include "SchemaProvider.h"

#include "Shared/mapd_shared_mutex.h"

class SimpleSchemaProvider : public SchemaProvider {
 public:
  SimpleSchemaProvider(hdk::ir::Context& ctx, int id, const std::string& name)
      : ctx_(ctx), id_(id), name_(name) {}
  ~SimpleSchemaProvider() override = default;

  int getId() const override { return id_; }
  std::string_view getName() const override { return name_; }

  std::vector<int> listDatabases() const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    std::vector<int> res;
    std::unordered_set<int> ids;
    for (auto& pr : table_index_by_name_) {
      if (!ids.count(pr.first)) {
        res.push_back(pr.first);
        ids.insert(pr.first);
      }
    }
    return res;
  }

  TableInfoList listTables(int db_id) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    TableInfoList res;
    if (table_index_by_name_.count(db_id)) {
      for (auto& pr : table_index_by_name_.at(db_id)) {
        res.push_back(pr.second);
      }
    }
    return res;
  }

  ColumnInfoList listColumns(int db_id, int table_id) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    return listColumnsNoLock(db_id, table_id);
  }

  using SchemaProvider::listColumns;

  TableInfoPtr getTableInfo(int db_id, int table_id) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    return getTableInfoNoLock(db_id, table_id);
  }

  TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    return getTableInfoNoLock(db_id, table_name);
  }

  using SchemaProvider::getTableInfo;

  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    auto it = column_infos_.find({db_id, table_id, col_id});
    if (it != column_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  ColumnInfoPtr getColumnInfo(int db_id,
                              int table_id,
                              const std::string& col_name) const override {
    mapd_shared_lock<mapd_shared_mutex> lock(schema_mutex_);
    auto table_it = column_index_by_name_.find({db_id, table_id});
    if (table_it != column_index_by_name_.end()) {
      auto col_it = table_it->second.find(col_name);
      if (col_it != table_it->second.end()) {
        return col_it->second;
      }
    }
    return nullptr;
  }

  using SchemaProvider::getColumnInfo;

 protected:
  TableInfoPtr getTableInfoNoLock(int db_id, int table_id) const {
    auto it = table_infos_.find({db_id, table_id});
    if (it != table_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  TableInfoPtr getTableInfoNoLock(int db_id, const std::string& table_name) const {
    auto db_it = table_index_by_name_.find(db_id);
    if (db_it != table_index_by_name_.end()) {
      auto table_it = db_it->second.find(table_name);
      if (table_it != db_it->second.end()) {
        return table_it->second;
      }
    }
    return nullptr;
  }

  ColumnInfoList listColumnsNoLock(int db_id, int table_id) const {
    CHECK_EQ(column_index_by_name_.count({db_id, table_id}), (size_t)1);
    auto& table_cols = column_index_by_name_.at({db_id, table_id});
    ColumnInfoList res;
    res.reserve(table_cols.size());
    for (auto [col_name, col_info] : table_cols) {
      res.push_back(col_info);
    }
    std::sort(res.begin(), res.end(), [](ColumnInfoPtr& lhs, ColumnInfoPtr& rhs) -> bool {
      return lhs->column_id < rhs->column_id;
    });
    return res;
  }

  ColumnInfoList listColumnsNoLock(const TableRef& ref) const {
    return listColumnsNoLock(ref.db_id, ref.table_id);
  }

  TableInfoPtr addTableInfo(TableInfoPtr table_info) {
    table_infos_[*table_info] = table_info;
    table_index_by_name_[table_info->db_id][table_info->name] = table_info;
    return table_info;
  }

  template <typename... Ts>
  TableInfoPtr addTableInfo(Ts... args) {
    return addTableInfo(std::make_shared<TableInfo>(args...));
  }

  ColumnInfoPtr addColumnInfo(ColumnInfoPtr col_info) {
    column_infos_[*col_info] = col_info;
    column_index_by_name_[{col_info->db_id, col_info->table_id}][col_info->name] =
        col_info;
    return col_info;
  }

  template <typename... Ts>
  ColumnInfoPtr addColumnInfo(Ts... args) {
    return addColumnInfo(std::make_shared<ColumnInfo>(args...));
  }

  ColumnInfoPtr addRowidColumn(int db_id, int table_id, int col_id) {
    CHECK_EQ(column_index_by_name_.count({db_id, table_id}), (size_t)1);
    return addColumnInfo(db_id, table_id, col_id, "rowid", ctx_.int64(), true);
  }

  void dropTable(int db_id, int table_id) {
    auto tinfo = getTableInfoNoLock(db_id, table_id);
    CHECK(tinfo);
    auto col_infos = listColumnsNoLock(*tinfo);
    table_infos_.erase(*tinfo);
    table_index_by_name_.at(db_id).erase(tinfo->name);
    for (auto& col_info : col_infos) {
      column_infos_.erase(*col_info);
      column_index_by_name_.at(*tinfo).erase(col_info->name);
    }
  }

  using TableByNameMap = std::unordered_map<std::string, TableInfoPtr>;
  using ColumnByNameMap = std::unordered_map<std::string, ColumnInfoPtr>;

  hdk::ir::Context& ctx_;
  int id_;
  std::string name_;
  TableInfoMap table_infos_;
  std::unordered_map<int, TableByNameMap> table_index_by_name_;
  ColumnInfoMap column_infos_;
  std::unordered_map<TableRef, ColumnByNameMap> column_index_by_name_;
  mutable mapd_shared_mutex schema_mutex_;
};
