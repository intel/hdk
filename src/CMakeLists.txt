include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

set(ANALYZER_FILES ../omniscidb/Analyzer/Analyzer.cpp ../omniscidb/QueryEngine/DateTruncate.cpp)

add_library(HDK HDK.h HDK.cpp ${ANALYZER_FILES}) 
target_link_libraries(HDK glog::glog)

