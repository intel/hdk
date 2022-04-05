set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=hdk)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=hdk)
add_public_tablegen_target(HDKOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(HDKTypesIncGen)

