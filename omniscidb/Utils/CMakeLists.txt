set(utils_source_files
    ChunkIter.cpp
    ExtractFromTime.cpp
    ExtractStringFromTime.cpp
    Regexp.cpp
    StringLike.cpp
)

add_library(Utils ${utils_source_files})
add_dependencies(Utils Calcite)
target_link_libraries(Utils OSDependent ${Boost_LIBRARIES} Shared)
