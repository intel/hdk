set(result_set_registry_source_files
    ColumnarResults.cpp
    ResultSetMetadata.cpp
    ResultSetRegistry.cpp
    ResultSetTableToken.cpp
)

add_library(ResultSetRegistry ${result_set_registry_source_files})

target_link_libraries(ResultSetRegistry SchemaMgr ResultSet StringDictionary DataMgr IR Shared Logger Utils)
