include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

set(ANALYZER_FILES ../omniscidb/Analyzer/Analyzer.cpp ../omniscidb/QueryEngine/DateTruncate.cpp)

add_subdirectory(mlir)

include_directories(${CMAKE_BINARY_DIR}/src/mlir)

# may need to add explicit depends here on tablegen
set(DIALECT_FILES mlir/Dialect.cpp mlir/Ops.cpp)

add_library(HDK HDK.h HDK.cpp ${DIALECT_FILES} ${ANALYZER_FILES})
add_dependencies(HDK HDKOpsIncGen)
target_link_libraries(HDK Logger Shared MLIRIR)

