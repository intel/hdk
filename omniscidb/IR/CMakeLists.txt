set(ir_source_files
    Context.cpp
    Context.h
    DateTime.cpp
    Datum.cpp
    Exception.h
    Expr.cpp
    Expr.h
    Type.cpp
    Type.h
    TypeUtils.cpp
    TypeUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
)

add_library(IR ${ir_source_files})

target_link_libraries(IR Shared)
