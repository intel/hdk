#include "HDK.h"

#include <iostream>
#include <memory>

int main(void) {
  int table_id = 0;
  SQLTypeInfo sql_type(SQLTypes::kINT);

  const auto col_expr =
      makeExpr<Analyzer::ColumnVar>(sql_type, table_id, /*column_id=*/0, 0);

  const auto work_unit = hdk::create_work_unit();

  std::cout << "Test program worked: " << col_expr->toString() << std::endl;
}
