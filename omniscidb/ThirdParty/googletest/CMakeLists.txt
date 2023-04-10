include_directories(.)
add_library(gtest gmock-gtest-all.cc)

find_package(Threads REQUIRED)
target_link_libraries(gtest PRIVATE Threads::Threads)
