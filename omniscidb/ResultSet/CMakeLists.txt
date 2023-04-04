set(result_set_source_files
    ColSlotContext.cpp
    QueryMemoryDescriptor.cpp
    ResultSet.cpp
    ResultSetIteration.cpp
    ResultSetStorage.cpp
    RowSetMemoryOwner.cpp
    StreamingTopN.cpp
)

add_library(ResultSet ${result_set_source_files})

target_link_libraries(ResultSet DataMgr IR Logger Utils StringDictionary ${Folly_LIBRARIES})
