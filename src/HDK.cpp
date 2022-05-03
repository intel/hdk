#include "HDK.h"

#include <iostream>

#include "AST/AST.h"
#include "MLIRGen.h"

#include "mlir/Dialect.h"
#include "mlir/InitAllPasses.h"

void mlir_test() {
  std::cout << "Initializing MLIR" << std::endl;
  //  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  //  registry.insert<hdk::HDKDialect>(); // is loaded below?
  registry.insert<mlir::StandardOpsDialect>();

  std::cout << "MLIR Initialized" << std::endl;

  std::cout << "### Testing MLIR Dialect ###" << std::endl;

  hdk::AST::KernelSequence sequence;
  auto kernel = hdk::AST::Kernel();
  kernel.projected_expressions.emplace_back(
      std::make_unique<hdk::AST::Constant>(SQLTypes::kINT, /*notnull=*/false));
  sequence.emplace_back(std::move(kernel));

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<hdk::HDKDialect>();
  auto mlir_module = hdk::mlirGen(context, sequence);
  if (!mlir_module) {
    std::cerr << "Failed to generate MLIR module!" << std::endl;
  } else {
    mlir_module->dump();
  }
}