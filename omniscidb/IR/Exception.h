/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <exception>
#include <sstream>
#include <string>

namespace hdk::ir {

class Error : public std::exception {
 public:
  Error() {}
  Error(std::string desc) : desc_(std::move(desc)) {}

  const char* what() const noexcept override { return desc_.c_str(); }

  void appendDesc(const std::string& str) const { desc_ += str; }

 private:
  mutable std::string desc_;
};

class TypeError : public Error {
 public:
  TypeError() {}
  TypeError(std::string desc) : Error(std::move(desc)) {}
};

class InvalidTypeError : public TypeError {
 public:
  InvalidTypeError() {}
  InvalidTypeError(std::string desc) : TypeError(std::move(desc)) {}
};

class UnsupportedTypeError : public TypeError {
 public:
  UnsupportedTypeError() {}
  UnsupportedTypeError(std::string desc) : TypeError(std::move(desc)) {}
};

template <typename ErrorType, typename T>
inline typename std::enable_if<std::is_base_of<hdk::ir::Error, ErrorType>::value,
                               const ErrorType&>::type
operator<<(const ErrorType& error, const T& v) {
  std::stringstream ss;
  ss << v;
  error.appendDesc(ss.str());
  return error;
}

}  // namespace hdk::ir
