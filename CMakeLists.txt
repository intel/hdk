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

# OmniSciDB submodule
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

add_subdirectory(omniscidb/Shared)

add_subdirectory(omniscidb/OSDependent)

# TODO: replace with glog
include_directories(omniscidb/ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)

add_subdirectory(omniscidb/Logger)


# Source
add_subdirectory(src)

add_executable(TestDriver apps/TestDriver.cpp)

target_link_libraries(TestDriver PRIVATE HDK)
target_link_libraries(TestDriver PRIVATE ${Arrow_LIBRARIES} Shared OSDependent Logger ${Boost_LIBRARIES})

target_include_directories(TestDriver PRIVATE src/)
