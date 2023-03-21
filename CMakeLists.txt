cmake_minimum_required(VERSION 3.16.3)

set(HDK_VERSION_MAJOR "0")
set(HDK_VERSION_MINOR "4")
set(HDK_VERSION_PATCH "0")
set(HDK_VERSION_EXTRA "dev")
set(HDK_VERSION_RAW_NUMERIC "${HDK_VERSION_MAJOR}.${HDK_VERSION_MINOR}.${HDK_VERSION_PATCH}")
set(HDK_VERSION_RAW "${HDK_VERSION_MAJOR}.${HDK_VERSION_MINOR}.${HDK_VERSION_PATCH}${HDK_VERSION_EXTRA}")

project(
        HeterogeneousDataKernels
        VERSION ${HDK_VERSION_RAW_NUMERIC}
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force -O0 optimization level for debug builds.
if (WIN32)
  # Export all symbols from DLLs by default. It is necessary because there are too many of them
  # to mark explicitly.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # 0x0600 means Windows Vista (anything after XP). This setting is needed because without it
  # boost defines different API levels for different parts of the library and functions appear in
  # different namespaces.
  add_definitions("-DBOOST_USE_WINAPI_VERSION=0x0600")
  # On windows TBB adds tbb12_debug.lib unless the following define is set. This tbb12_debug.lib
  # is not present in conda-forge tbb build.
  add_definitions("-D__TBB_NO_IMPLICIT_LINKAGE")
  # Always use multithreaded DLL runtime (not debug) because using debug requires other dependencies
  # to be built with debug runtime which we don't have.
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
  # Add a special option to require MSVC to define __cplusplus macro to a proper value
  # corresponding to the real C++ language standard version. Without this option Microsoft compiler
  # always defines __cplusplus to 199711L. A valid number is needed in llvm headers.
  add_compile_options(/Zc:__cplusplus)
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

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

# External Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/omniscidb/cmake/Modules")

set(ENABLE_CONDA OFF)
if(DEFINED ENV{CONDA_PREFIX})
  set(ENABLE_CONDA ON)
  set(CMAKE_SYSROOT "$ENV{CONDA_BUILD_SYSROOT}")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
endif()

string(TIMESTAMP HDK_BUILD_DATE "%Y%m%d")
find_package(Git)
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --short=10 HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE HDK_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
file(WRITE ${CMAKE_BINARY_DIR}/HDK_GIT_HASH.txt "${HDK_GIT_HASH}\n")
file(STRINGS ${CMAKE_BINARY_DIR}/HDK_GIT_HASH.txt HDK_GIT_HASH)


configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/release.h"
  "${CMAKE_BINARY_DIR}/MapDRelease.h"
  @ONLY
  )
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/HDK_GIT_HASH.txt)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MapDRelease.h)
  # required to force regen of HDK_GIT_HASH.txt, MapDRelease.h
  add_custom_target(rerun_cmake ALL
    COMMAND cmake .
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
include_directories(${CMAKE_BINARY_DIR})

# Folly
option(ENABLE_FOLLY "Use Folly" ON)
if(ENABLE_FOLLY)
  find_package(Folly)
  if(NOT Folly_FOUND)
    set(ENABLE_FOLLY OFF CACHE BOOL "Use Folly" FORCE)
  else()
    set(FOLLY_LIBRARIES "")
    add_definitions("-DHAVE_FOLLY")
    list(APPEND Folly_LIBRARIES Folly::folly ${Glog_LIBRARIES})
    # TODO: use Folly::folly_deps?
    if(MSVC)
      find_package(Libevent COMPONENTS core REQUIRED)
      list(APPEND Folly_LIBRARIES libevent::core)
    endif()
  endif()
endif()


# SQLite
include_directories(omniscidb/ThirdParty/sqlite3)
add_subdirectory(omniscidb/ThirdParty/sqlite3)

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

# Google log
add_subdirectory(third_party/glog-0.5.0 EXCLUDE_FROM_ALL)
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

# Dwarf Bench
option(ENABLE_DWARF_BENCH "Enable DwarfBench library for Cost Model" OFF)
if(ENABLE_DWARF_BENCH)
  find_package(dbench REQUIRED)
  message(STATUS "Dwarf Bench enabled ${dbench_DIR}")
  add_definitions("-DENABLE_DWARF_BENCH")
endif()

option(ENABLE_L0 "Enable level zero support" OFF)
if(ENABLE_L0)
  find_package(LevelZero REQUIRED COMPONENTS ${LevelZero_COMPONENTS})
  include_directories(${LevelZero_INCLUDE_DIRS})
  add_definitions("-DHAVE_L0")
endif()

set(CUDA_USE_STATIC_CUDA_RUNTIME ON CACHE STRING "Use static CUDA runtime")

option(ENABLE_CUDA "Enable CUDA support" OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # A temporary workaround for non-conda envs
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  list(APPEND CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver)
  add_definitions("-DHAVE_CUDA")

else()
  set(CUDA_LIBRARIES "")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-cpu")
endif()

# CUDA architecture flags
if("${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug")
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" ON)
else()
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" OFF)
endif()
if(ENABLE_CUDA)
  set(MAPD_CUDA_OPTIONS)
  # Set Thrust debug mode for CUDA compilation project-wide
  string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)
  if(CMAKE_BUILD_TYPE_UPPERCASE MATCHES DEBUG)
    list(APPEND MAPD_CUDA_OPTIONS -DTHRUST_DEBUG --debug)
  else()
    list(APPEND MAPD_CUDA_OPTIONS -O3)
  endif()

  list(APPEND MAPD_CUDA_OPTIONS -Xcompiler -fPIC -D_FORCE_INLINES -std=c++17)

  if(ENABLE_ONLY_ONE_ARCH)
    execute_process(
      COMMAND cmake -S ${CMAKE_SOURCE_DIR}/NvidiaComputeCapability -B NvidiaComputeCapability
      OUTPUT_QUIET
      ERROR_QUIET
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    execute_process(
      COMMAND cmake --build NvidiaComputeCapability
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.out.txt
      ERROR_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.err.txt
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set(NVIDIA_COMPUTE_CAPABILITY "")
    if (EXISTS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt)
      file(STRINGS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt NVIDIA_COMPUTE_CAPABILITY)
    endif()
  endif()
  if (ENABLE_ONLY_ONE_ARCH AND NOT "${NVIDIA_COMPUTE_CAPABILITY}" STREQUAL "")
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
      set(CMAKE_CUDA_ARCHITECTURES ${NVIDIA_COMPUTE_CAPABILITY}-virtual)
      list(APPEND MAPD_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
      message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      set (CUDA_COMPILATION_ARCH
        -gencode=arch=compute_${NVIDIA_COMPUTE_CAPABILITY},code=compute_${NVIDIA_COMPUTE_CAPABILITY}
        -Wno-deprecated-gpu-targets
      )
      message(STATUS "CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILATION_ARCH}>")
    endif()
    add_custom_target(clean_nvidia_compute_capability
      COMMAND ${CMAKE_BUILD_TOOL} clean
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/NvidiaComputeCapability
    )
    add_dependencies(clean-all clean_nvidia_compute_capability)
  else()
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
      message(STATUS "CMake 3.18+, Setting CUDA_ARCHITECTURES.")
      set(CMAKE_CUDA_ARCHITECTURES
          50-virtual
          60-virtual
          70-virtual
          75-virtual
          80-virtual)
      list(APPEND MAPD_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
      message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      message(STATUS "CMake 3.17 or under, setting CUDA architecture flags manually.")
      set(CUDA_COMPILATION_ARCH
        -gencode=arch=compute_50,code=compute_50;
        -gencode=arch=compute_60,code=compute_60;
        -gencode=arch=compute_70,code=compute_70;
        -gencode=arch=compute_75,code=compute_75;
        -gencode=arch=compute_80,code=compute_80;
        -Wno-deprecated-gpu-targets)
      message(STATUS "CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
      list(APPEND MAPD_CUDA_OPTIONS ${CUDA_COMPILATION_ARCH})
    endif()
    if(ENABLE_ONLY_ONE_ARCH)
      message(STATUS "ENABLE_ONLY_ONE_ARCH ignored because NvidiaComputeCapability.txt not found or not readable")
    endif()
  endif()
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    include(ProcessorCount)
    ProcessorCount(N)
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 11.3 AND NOT N EQUAL 0)
      message(STATUS "Enabling NVCC multi-threaded compilation with ${N} threads.")
      list(APPEND MAPD_CUDA_OPTIONS --threads ${N})
      set(NVCC_THREADS --threads ${N})
    endif()
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${MAPD_CUDA_OPTIONS}>")
endif()

if (ENABLE_CUDA)
  list(INSERT Arrow_LIBRARIES 0 ${Arrow_GPU_CUDA_LIBRARIES})
endif()

# LLVM
find_package(LLVM CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_library(CLANG_LIB clang-cpp)
find_library(LLVM_LIB LLVM)
# Deps builds use separate libs for each clang component, while some distros now bundle into a single lib
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" OR NOT LLVM_LIB)
  set(LLVM_COMPONENTS support orcjit core irreader option linker)

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

# SPIRV-Translator
if(ENABLE_L0)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(LLVMSPIRVLib REQUIRED IMPORTED_TARGET LLVMSPIRVLib)
  message(STATUS "Found LLVMSPIRVLib of version ${LLVMSPIRVLib_VERSION}")
  list(PREPEND llvm_libs PkgConfig::LLVMSPIRVLib)
endif()

# address and thread sanitizer
option(ENABLE_STANDALONE_CALCITE "Require standalone Calcite server" OFF)
option(ENABLE_ASAN "Enable address sanitizer" OFF)
option(ENABLE_TSAN "Enable thread sanitizer" OFF)
option(ENABLE_UBSAN "Enable undefined behavior sanitizer" OFF)
if(ENABLE_ASAN)
  set(SAN_FLAGS "-fsanitize=address -O1 -fno-omit-frame-pointer")
  add_definitions("-DWITH_DECODERS_BOUNDS_CHECKING")
elseif(ENABLE_TSAN)
  add_definitions("-DHAVE_TSAN")
  # Copy the config directory to the build dir for TSAN suppressions
  file(COPY omniscidb/config DESTINATION ${CMAKE_BINARY_DIR})

  set(SAN_FLAGS "-fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer")
  # required for older GCC, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64354
  add_definitions("-D__SANITIZE_THREAD__")
elseif(ENABLE_UBSAN)
  set(SAN_FLAGS "-fsanitize=undefined -fPIC -O1 -fno-omit-frame-pointer")
endif()
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
  set(ENABLE_STANDALONE_CALCITE ON)
endif()

option(ENABLE_DECODERS_BOUNDS_CHECKING "Enable bounds checking for column decoding" OFF)

if(ENABLE_STANDALONE_CALCITE)
  add_definitions("-DSTANDALONE_CALCITE")
endif()

# OmniSciDB submodule
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

add_subdirectory(omniscidb/Shared)

add_subdirectory(omniscidb/OSDependent)

include_directories(omniscidb/ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)
if(NOT MSVC)
  # At the present time the current vcpkg version of rapidjson is 2020-09-14:
  # https://github.com/microsoft/vcpkg/blob/master/versions/r-/rapidjson.json
  # and the Windows build fails because it does not have this fix:
  # https://github.com/Tencent/rapidjson/pull/1568
  # Once vcpkg's rapidjson has this fix then let's try not making this exception for MSVC.
  # When this changes, remove this exception from all other similar CMakeLists.txt files too.
  add_definitions(-DRAPIDJSON_NOMEMBERITERATORCLASS)
endif()
include_directories(omniscidb/ThirdParty/googletest)
add_subdirectory(omniscidb/ThirdParty/googletest)

# TODO: replace with glog
add_subdirectory(omniscidb/Logger)

add_subdirectory(omniscidb/Utils)
add_subdirectory(omniscidb/Calcite)
add_subdirectory(omniscidb/ConfigBuilder)
add_subdirectory(omniscidb/SchemaMgr)
add_subdirectory(omniscidb/StringDictionary)
add_subdirectory(omniscidb/L0Mgr)
add_subdirectory(omniscidb/CudaMgr)
add_subdirectory(omniscidb/DataMgr)
add_subdirectory(omniscidb/ArrowStorage)
add_subdirectory(omniscidb/Analyzer)
add_subdirectory(omniscidb/IR)
add_subdirectory(omniscidb/SqliteConnector)
add_subdirectory(omniscidb/QueryBuilder)
add_subdirectory(omniscidb/QueryEngine)

# Source
add_subdirectory(src)

if(BUILD_SHARED_LIBS AND ENABLE_PYTHON)
  add_subdirectory(python)
endif()

install(TARGETS OSDependent Logger Shared Utils Calcite ArrowStorage StringDictionary DataMgr CudaMgr SchemaMgr L0Mgr QueryBuilder QueryEngine Analyzer IR ConfigBuilder SqliteConnector RUNTIME)
if(ENABLE_CUDA)
  install(FILES ${CMAKE_BINARY_DIR}/omniscidb/QueryEngine/cuda_mapd_rt.fatbin DESTINATION QueryEngine COMPONENT "exe")
endif()


add_executable(TestDriver apps/TestDriver.cpp)

target_link_libraries(TestDriver PRIVATE HDK)
target_link_libraries(TestDriver PRIVATE ${Arrow_LIBRARIES} IR QueryEngine StringDictionary Analyzer Shared OSDependent Logger ${llvm_libs} ${Boost_LIBRARIES})

target_include_directories(TestDriver PRIVATE src/)

add_custom_target(clean-all
  COMMAND ${CMAKE_BUILD_TOOL} clean
 )

file(GLOB_RECURSE GENERATED_PYTHON_CPP ${CMAKE_SOURCE_DIR}/python/**/*.cpp)
add_custom_target(hdk_python_clean
  COMMAND  ${CMAKE_COMMAND} -E remove ${GENERATED_PYTHON_CPP}
)

add_dependencies(clean-all hdk_python_clean)
add_dependencies(clean-all calcite_java_clean)


option(ENABLE_TESTS "Build unit tests" ON)
if (ENABLE_TESTS)
  enable_testing()
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
  add_subdirectory(omniscidb/ThirdParty/googlebenchmark)
  add_subdirectory(omniscidb/UdfCompiler)
  add_subdirectory(omniscidb/Tests/ArrowSQLRunner)
  add_subdirectory(omniscidb/Tests)
endif()

option(ENABLE_BENCHMARKS "Build benchmarks" ON)
if (ENABLE_TESTS AND ENABLE_BENCHMARKS)
  add_subdirectory(omniscidb/Benchmarks/taxi)
endif()

