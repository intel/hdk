include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/omniscidb)

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
    set(LLVM_COMPONENTS support mcjit core irreader option linker)

    if (ENABLE_INTEL_JIT_LISTENER)
        list(APPEND LLVM_COMPONENTS inteljitevents)
    endif ()

    llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} ${LLVM_COMPONENTS})
else ()
    set(clang_libs ${CLANG_LIB})
    set(llvm_libs ${LLVM_LIB})
endif ()

# for generating LLVM IR for execution engine runtime
find_program(llvm_clangpp_cmd NAME clang++ HINTS ${LLVM_TOOLS_BINARY_DIR})

# TBB
find_package(TBB REQUIRED)
message(STATUS "TBB library is found with ${TBB_DIR}")
add_definitions("-DHAVE_TBB")
add_definitions("-DTBB_PREVIEW_TASK_GROUP_EXTENSIONS=1")
set(ENABLE_TBB ON)
add_definitions("-DENABLE_TBB")

set(ENABLE_CUDA off)

set(ANALYZER_FILES ../omniscidb/Analyzer/Analyzer.cpp ../omniscidb/QueryEngine/DateTruncate.cpp)

set(EXECUTION_ENGINE_FILES ExecutionEngine/ExecutionEngine.cpp
        ../omniscidb/Fragmenter/InsertOrderFragmenter.cpp
        ../omniscidb/QueryEngine/CgenState.cpp
        ../omniscidb/QueryEngine/Descriptors/ColSlotContext.cpp
        ../omniscidb/QueryEngine/Descriptors/QueryMemoryDescriptor.cpp
        ../omniscidb/QueryEngine/Execute.cpp
        ../omniscidb/QueryEngine/ExpressionRange.cpp
        ../omniscidb/QueryEngine/InputMetadata.cpp
        ../omniscidb/QueryEngine/InValuesBitmap.cpp
        ../omniscidb/QueryEngine/JoinHashTable/BaselineJoinHashTable.cpp
        ../omniscidb/QueryEngine/JoinHashTable/HashJoin.cpp
        ../omniscidb/QueryEngine/JoinHashTable/HashTable.cpp
        ../omniscidb/QueryEngine/JoinHashTable/OverlapsJoinHashTable.cpp
        ../omniscidb/QueryEngine/JoinHashTable/PerfectJoinHashTable.cpp
        ../omniscidb/QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.cpp
        ../omniscidb/QueryEngine/JoinHashTable/RangeJoinHashTable.cpp
        ../omniscidb/QueryEngine/LoopControlFlow/JoinLoop.cpp
        ../omniscidb/QueryEngine/NativeCodegen.cpp
        ../omniscidb/QueryEngine/PlanState.cpp
        ../omniscidb/QueryEngine/ResultSet.cpp
        ../omniscidb/QueryEngine/ResultSetIteration.cpp
        ../omniscidb/QueryEngine/ResultSetReduction.cpp
        ../omniscidb/QueryEngine/ResultSetReductionCodegen.cpp
        ../omniscidb/QueryEngine/ResultSetReductionJIT.cpp
        ../omniscidb/QueryEngine/ResultSetStorage.cpp
        ../omniscidb/QueryEngine/StringDictionaryGenerations.cpp
        ../omniscidb/QueryEngine/TableGenerations.cpp
        ../omniscidb/StringDictionary/StringDictionary.cpp
        ../omniscidb/StringDictionary/StringDictionaryProxy.cpp
        )

add_library(HDK HDK.h HDK.cpp ${ANALYZER_FILES} ${EXECUTION_ENGINE_FILES})
target_link_libraries(HDK glog::glog ${llvm_libs} TBB::tbb)

