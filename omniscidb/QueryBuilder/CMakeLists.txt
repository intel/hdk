if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
endif()

set(query_builder_source_files
    QueryBuilder.cpp
    QueryBuilder.h
)

add_library(QueryBuilder ${query_builder_source_files})

target_link_libraries(QueryBuilder IR Analyzer SchemaMgr Shared)
