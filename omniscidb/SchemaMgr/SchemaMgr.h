/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SchemaProvider.h"

#pragma once

class SchemaMgr : public SchemaProvider {
 public:
  ~SchemaMgr() = default;

  int getId() const override { return -1; };
  std::string_view getName() const override { return "SchemaMgr"; };

  std::vector<int> listDatabases() const override;

  TableInfoList listTables(int db_id) const override;

  ColumnInfoList listColumns(int db_id, int table_id) const override;

  TableInfoPtr getTableInfo(int db_id, int table_id) const override;
  TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const override;

  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override;
  ColumnInfoPtr getColumnInfo(int db_id,
                              int table_id,
                              const std::string& col_name) const override;

  void registerProvider(int schema_id, SchemaProviderPtr schema_provider);

 protected:
  const SchemaProvider* getMgr(int db_id) const;

  std::unordered_map<int, SchemaProviderPtr> mgr_by_schema_id_;
};

using SchemaMgrPtr = std::shared_ptr<SchemaMgr>;

SchemaProviderPtr mergeProviders(const std::vector<SchemaProviderPtr>& providers);
