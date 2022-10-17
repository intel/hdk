include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

add_library(HDK HDK.h HDK.cpp) 
target_link_libraries(HDK glog::glog)

