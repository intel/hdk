set(udf_compiler_source_files
    UdfCompiler.cpp)

add_definitions("-DUDF_INCLUDE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/..\"")

# Suppress multi-line comment warning which is not disabled by a diagnostic pragma
# for some reason.
if(NOT WIN32)
    set_source_files_properties(UdfCompiler.cpp PROPERTIES COMPILE_OPTIONS "-Wno-comment")
endif()

add_library(UdfCompiler ${udf_compiler_source_files})

# Clang
find_package(Clang REQUIRED)
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})

target_link_libraries(UdfCompiler Logger ${clang_libs})

