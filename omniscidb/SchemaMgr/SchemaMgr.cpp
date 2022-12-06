/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SchemaMgr.h"

std::vector<int> SchemaMgr::listDatabases() const {
  std::vector<int> res;
  for (auto& pr : mgr_by_schema_id_) {
    auto mgr_res = pr.second->listDatabases();
    res.insert(res.end(), mgr_res.begin(), mgr_res.end());
  }
  return res;
}

TableInfoList SchemaMgr::listTables(int db_id) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->listTables(db_id);
  }
  return {};
}

ColumnInfoList SchemaMgr::listColumns(int db_id, int table_id) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->listColumns(db_id, table_id);
  }
  return {};
}

TableInfoPtr SchemaMgr::getTableInfo(int db_id, int table_id) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->getTableInfo(db_id, table_id);
  }
  return nullptr;
}
TableInfoPtr SchemaMgr::getTableInfo(int db_id, const std::string& table_name) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->getTableInfo(db_id, table_name);
  }
  return nullptr;
}

ColumnInfoPtr SchemaMgr::getColumnInfo(int db_id, int table_id, int col_id) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->getColumnInfo(db_id, table_id, col_id);
  }
  return nullptr;
}
ColumnInfoPtr SchemaMgr::getColumnInfo(int db_id,
                                       int table_id,
                                       const std::string& col_name) const {
  auto mgr = getMgr(db_id);
  if (mgr) {
    return mgr->getColumnInfo(db_id, table_id, col_name);
  }
  return nullptr;
}

void SchemaMgr::registerProvider(int schema_id, SchemaProviderPtr schema_provider) {
  CHECK_GE(schema_id, MIN_SCHEMA_ID);
  CHECK_LE(schema_id, MAX_SCHEMA_ID);
  if (mgr_by_schema_id_.count(schema_id)) {
    throw std::runtime_error("Detected schema provider with duplicated schema id: " +
                             std::to_string(schema_id));
  }
  mgr_by_schema_id_.emplace(schema_id, schema_provider);
}

const SchemaProvider* SchemaMgr::getMgr(int db_id) const {
  auto it = mgr_by_schema_id_.find(getSchemaId(db_id));
  if (it != mgr_by_schema_id_.end()) {
    return it->second.get();
  }
  return nullptr;
}