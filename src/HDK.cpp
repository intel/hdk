#include "HDK.h"

#include "../omniscidb/QueryEngine/Execute.h"

namespace hdk {

WorkUnit create_work_unit() {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  const JoinQualsPerNestingLevel join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  std::vector<Analyzer::Expr*> target_exprs;
  const std::shared_ptr<Analyzer::Estimator> estimator;
  const SortInfo sort_info{{}, SortAlgorithm::Default, 0, 0};
  size_t scan_limit;
  RegisteredQueryHint query_hint;

  auto ra_exe_unit = RelAlgExecutionUnit{input_descs,
                                         input_col_descs,
                                         simple_quals,
                                         quals,
                                         join_quals,
                                         groupby_exprs,
                                         target_exprs,
                                         estimator,
                                         sort_info,
                                         scan_limit,
                                         query_hint};

  return WorkUnit{};
}

}
