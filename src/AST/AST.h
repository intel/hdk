#include <glog/logging.h>

#include "Analyzer/Analyzer.h"

#include <vector>
#include <memory>
#include <string>

namespace hdk {
namespace AST {

using Expr = Analyzer::Expr;

struct Kernel {
  std::vector<Expr> projected_expressions;
};

using KernelSequence = std::vector<Kernel>;

}
}  // namespace hdk
