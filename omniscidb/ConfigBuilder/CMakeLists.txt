add_library(ConfigBuilder ConfigBuilder.cpp)

target_link_libraries(ConfigBuilder Logger Shared ${Boost_LIBRARIES})
