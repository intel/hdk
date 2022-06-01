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

set(ENABLE_CONDA OFF)
if(DEFINED ENV{CONDA_PREFIX})
  set(ENABLE_CONDA ON)
  set(CMAKE_SYSROOT "$ENV{CONDA_BUILD_SYSROOT}")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_PYTHON "Build Python libraries" ON)

if(BUILD_SHARED_LIBS)
  add_definitions("-DENABLE_SHARED_LIBS")
  # With no this option all installed shared objects would get an empty
  # rpath that would break a link with libjvm.so.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# Copy ThirdParty to build dir so OmniSciDB dependencies can be copied over. Note that third_party is available internally for HDK specific dependencies.
file(COPY "${CMAKE_SOURCE_DIR}/ThirdParty" DESTINATION "${CMAKE_BINARY_DIR}/")

# External Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/omniscidb/cmake/Modules")

# Google log
add_subdirectory(third_party/glog-0.5.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/glog-0.5.0/cmake")
cmake_policy(SET CMP0024 OLD) # build glog with cmake bundled standard on 20.04
find_package(glog REQUIRED)

# Arrow
find_package(Arrow REQUIRED)
add_definitions("-DARROW_NO_DEPRECATED_API")
include_directories(${Arrow_INCLUDE_DIRS})

# Parquet
find_package(Parquet REQUIRED)

# Boost, required for OmniSciDB
add_definitions("-DBOOST_LOG_DYN_LINK") # dyn linking only
find_package(Boost COMPONENTS log log_setup filesystem program_options regex system thread timer locale iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# TBB
find_package(TBB REQUIRED)
add_definitions("-DENABLE_TBB")
add_definitions("-DHAVE_TBB")
add_definitions("-DTBB_PREVIEW_TASK_GROUP_EXTENSIONS=1")

# LLVM
find_library(CLANG_LIB clang-cpp)
find_library(LLVM_LIB LLVM)
# Deps builds use separate libs for each clang component, while some distros now bundle into a single lib
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" OR NOT LLVM_LIB)
  set(LLVM_COMPONENTS support mcjit core irreader option linker)

  if(ENABLE_INTEL_JIT_LISTENER)
    list(APPEND LLVM_COMPONENTS inteljitevents)
  endif()

  llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} ${LLVM_COMPONENTS})
  set(clang_libs
      clangFrontend
      clangSerialization
      clangDriver
      clangTooling
      clangParse
      clangSema
      clangAnalysis
      clangEdit
      clangAST
      clangLex
      clangBasic
      clangRewrite
      clangRewriteFrontend)

  # LLVMSupport explicitly lists tinfo in its INTERFACE_LINK_LIBRARIES, even
  # though we provide it in our build of ncurses. Since LLVMSupport is listed
  # as a requirement for other llvm libs, we need to walk through the entire
  # list in order to remove all instances of tinfo.
  foreach(lib ${llvm_libs})
    get_target_property(interface_libs ${lib} INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM interface_libs tinfo z rt pthread -lpthread m dl)
    set_target_properties(${lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
  endforeach()

  list(APPEND llvm_libs ${CURSES_NCURSES_LIBRARY})
else()
  if(NOT CLANG_LIB)
    message(FATAL_ERROR "Could not find CLANG library.")
  endif()

  set(clang_libs ${CLANG_LIB})
  set(llvm_libs ${LLVM_LIB})
endif()

# OmniSciDB submodule
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

add_subdirectory(omniscidb/Shared)

add_subdirectory(omniscidb/OSDependent)

include_directories(omniscidb/ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)
include_directories(omniscidb/ThirdParty/googletest)
add_subdirectory(omniscidb/ThirdParty/googletest)

# TODO: replace with glog
add_subdirectory(omniscidb/Logger)

add_subdirectory(omniscidb/Utils)
add_subdirectory(omniscidb/Calcite)
add_subdirectory(omniscidb/SchemaMgr)
add_subdirectory(omniscidb/StringDictionary)
add_subdirectory(omniscidb/L0Mgr)
add_subdirectory(omniscidb/CudaMgr)
add_subdirectory(omniscidb/DataMgr)
add_subdirectory(omniscidb/ArrowStorage)
add_subdirectory(omniscidb/Analyzer)
add_subdirectory(omniscidb/SqliteConnector)
add_subdirectory(omniscidb/QueryEngine)

# Source
add_subdirectory(src)

if(BUILD_SHARED_LIBS AND ENABLE_PYTHON)
  add_subdirectory(python)
endif()

install(TARGETS OSDependent Logger Shared Utils Calcite ArrowStorage StringDictionary DataMgr CudaMgr SchemaMgr L0Mgr QueryEngine Analyzer SqliteConnector RUNTIME)

add_executable(TestDriver apps/TestDriver.cpp)

target_link_libraries(TestDriver PRIVATE HDK)
target_link_libraries(TestDriver PRIVATE ${Arrow_LIBRARIES} Shared OSDependent Logger ${llvm_libs} ${Boost_LIBRARIES})

target_include_directories(TestDriver PRIVATE src/)
