/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Type.h"

#include "Shared/sqltypes.h"

#include <string>

namespace hdk::ir {

/*
 * @type TargetMetaInfo
 * @brief Encapsulates the name and the type of a relational projection.
 */
class TargetMetaInfo {
 public:
  TargetMetaInfo(const std::string& resname, const hdk::ir::Type* type)
      : resname_(resname), type_(type) {}
  const std::string& get_resname() const { return resname_; }
  const hdk::ir::Type* type() const { return type_; }

  std::string toString() const {
    return "TargetMetaInfo(" + resname_ + ", " + type_->toString() + ")";
  }

 private:
  std::string resname_;
  const hdk::ir::Type* type_;
};

inline std::ostream& operator<<(std::ostream& os, TargetMetaInfo const& tmi) {
  return os << "TargetMetaInfo(resname=" << tmi.get_resname()
            << " type=" << tmi.type()->toString() << ")";
}

}  // namespace hdk::ir
