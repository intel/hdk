cmake_minimum_required(VERSION 3.16.3)

project(
        HeterogeneousDataKernels
        VERSION 0.1
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set default build type to "Release w/ Debug Info" 
set(default_build_type "RelWithDebInfo")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Copy ThirdParty to build dir so OmniSciDB dependencies can be copied over. Note that third_party is available internally for HDK specific dependencies.
file(COPY "${CMAKE_SOURCE_DIR}/ThirdParty" DESTINATION "${CMAKE_BINARY_DIR}/")
file(COPY "${CMAKE_SOURCE_DIR}/omniscidb/ThirdParty/geo_samples" DESTINATION "${CMAKE_SOURCE_DIR}/ThirdParty")
file(COPY "${CMAKE_SOURCE_DIR}/omniscidb/ThirdParty/gdal-data" DESTINATION "${CMAKE_SOURCE_DIR}/ThirdParty")

# External Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

# Google log
add_subdirectory(third_party/glog-0.5.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/glog-0.5.0/cmake")
cmake_policy(SET CMP0024 OLD) # build glog with cmake bundled standard on 20.04
find_package(glog REQUIRED)

# Arrow
find_package(Arrow REQUIRED)
add_definitions("-DARROW_NO_DEPRECATED_API")
include_directories(${Arrow_INCLUDE_DIRS})

# Boost, required for OmniSciDB
add_definitions("-DBOOST_LOG_DYN_LINK") # dyn linking only
find_package(Boost COMPONENTS log log_setup filesystem program_options regex system thread timer locale iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# LLVM
find_package(LLVM CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_library(CLANG_LIB clang-cpp)
find_library(LLVM_LIB LLVM)

# TBB
find_package(TBB REQUIRED)
message(STATUS "TBB library is found with ${TBB_DIR}")
add_definitions("-DHAVE_TBB")
add_definitions("-DTBB_PREVIEW_TASK_GROUP_EXTENSIONS=1")
set(ENABLE_TBB ON)
add_definitions("-DENABLE_TBB")

# OmniSciDB submodule
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)
# Add OmniSciDB distributed subfolder for headers only
include_directories(omniscidb/Distributed/os)

add_subdirectory(omniscidb/Shared)

add_subdirectory(omniscidb/OSDependent)

set(MAPD_EDITION "OS")
string(TOLOWER "${MAPD_EDITION}" MAPD_EDITION_LOWER)
add_subdirectory(omniscidb/QueryEngine)

# TODO: replace with glog
include_directories(omniscidb/ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)

add_subdirectory(omniscidb/Logger)

# Source
add_subdirectory(src)

add_executable(TestDriver apps/TestDriver.cpp)

target_link_libraries(TestDriver PRIVATE HDK)
target_link_libraries(TestDriver PRIVATE ${Arrow_LIBRARIES} Shared OSDependent Logger ${Boost_LIBRARIES} QueryEngine ${TBB_LIBRARIES})

target_include_directories(TestDriver PRIVATE src/)
