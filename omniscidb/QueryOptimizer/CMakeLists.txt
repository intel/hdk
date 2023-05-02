set(query_optimizer_source_files
    CanonizeQuery.cpp
    CanonizeQuery.h
)

add_library(QueryOptimizer ${query_optimizer_source_files})

target_link_libraries(QueryOptimizer QueryBuilder IR Analyzer SchemaMgr Shared)
