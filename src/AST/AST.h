#pragma once

#include <glog/logging.h>

#include "Analyzer/Analyzer.h"

#include <memory>
#include <string>
#include <vector>

namespace hdk {
namespace AST {

using Expr = Analyzer::Expr;

struct Kernel {
  Kernel() {}

  std::vector<std::unique_ptr<Expr>> projected_expressions;
};

using Constant = Analyzer::Constant;

using KernelSequence = std::vector<Kernel>;

}  // namespace AST
}  // namespace hdk
