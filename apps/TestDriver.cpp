#include "HDK.h"

#include <iostream>
#include <memory>

int main(void) {
  auto input_data_type = std::make_unique<hdk::Int>();
  auto input_data = std::make_unique<hdk::Column>(std::move(input_data_type));
  auto output_type = std::make_unique<hdk::Int>();
  auto agg_expr = std::make_unique<hdk::Aggregate>(
      std::move(output_type), hdk::Aggregate::AggType::kCOUNT, std::move(input_data));

  std::cout << "Test program worked: " << agg_expr->toString() << std::endl;
}
