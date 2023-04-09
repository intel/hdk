set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes")
endif ()

set_source_files_properties(RuntimeFunctionsCodegenWithIncludes.cpp PROPERTIES COMPILE_FLAGS -O0)
set(query_engine_source_files
    AggregatedColRange.cpp
    ArithmeticIR.cpp
    ArrayIR.cpp
    ArrayOps.cpp
    ArrowResultSetConverter.cpp
    ArrowResultSet.cpp
    BitmapGenerators.cpp
    CalciteDeserializerUtils.cpp
    CardinalityEstimator.cpp
    CaseIR.cpp
    CastIR.cpp
    CgenState.cpp
    Codec.cpp
    ColRangeInfo.cpp
    ColumnFetcher.cpp
    ColumnIR.cpp
    CompareIR.cpp
    CompilationOptions.cpp
    Compiler/Backend.cpp
    Compiler/HelperFunctions.cpp
    ConstantIR.cpp
    DateTimeIR.cpp
    DateTimePlusRewrite.cpp
    DateTimeTranslator.cpp
    DateTruncate.cpp
    Descriptors/QueryCompilationDescriptor.cpp
    Descriptors/QueryFragmentDescriptor.cpp
    Descriptors/RelAlgExecutionDescriptor.cpp
    DeviceKernel.cpp
    Dispatchers/DefaultExecutionPolicy.cpp
    Dispatchers/RRExecutionPolicy.cpp
    Dispatchers/ProportionBasedExecutionPolicy.cpp
    EquiJoinCondition.cpp
    Execute.cpp
    ExecutionKernel.cpp
    ExpressionRange.cpp
    ExpressionRewrite.cpp
    ExtensionFunctionsBinding.cpp
    ExtensionFunctionsWhitelist.cpp
    ExtensionFunctions.ast
    ExtensionsIR.cpp
    ExternalExecutor.cpp
    FromTableReordering.cpp
    GpuInitGroupsImpl.cpp
    GpuInterrupt.cpp
    GpuMemUtils.cpp
    GpuSharedMemoryUtils.cpp
    InPlaceSort.cpp
    InValuesIR.cpp
    IRCodegen.cpp
    RowFuncBuilder.cpp
    InValuesBitmap.cpp
    InputMetadata.cpp
    JoinFilterPushDown.cpp
    JoinHashTable/BaselineJoinHashTable.cpp
    JoinHashTable/HashJoin.cpp
    JoinHashTable/HashTable.cpp
    JoinHashTable/PerfectJoinHashTable.cpp
    JoinHashTable/Runtime/HashJoinRuntime.cpp
    L0Kernel.cpp
    LogicalIR.cpp
    LLVMFunctionAttributesUtil.cpp
    LLVMGlobalContext.cpp
    MaxwellCodegenPatch.cpp
    MemoryLayoutBuilder.cpp
    MurmurHash.cpp
    NativeCodegen.cpp
    NvidiaKernel.cpp
    OutputBufferInitialization.cpp
    QueryPhysicalInputsCollector.cpp
    PlanState.cpp
    QueryRewrite.cpp
    QueryTemplateGenerator.cpp
    QueryExecutionContext.cpp
    QueryExecutionSequence.cpp
    QueryMemoryInitializer.cpp
    RelAlgDagBuilder.cpp
    RelAlgExecutor.cpp
    RelAlgTranslator.cpp
    RelAlgOptimizer.cpp
    RelAlgSchemaProvider.cpp
    ResultSetBuilder.cpp
    ResultSetReduction.cpp
    ResultSetReductionCodegen.cpp
    ResultSetReductionInterpreter.cpp
    ResultSetReductionInterpreterStubs.cpp
    ResultSetReductionJIT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LoopControlFlow/JoinLoop.cpp
    ResultSetSort.cpp
    RuntimeFunctions.cpp
    RuntimeFunctions.bc
    DynamicWatchdog.cpp
    ScalarCodeGenerator.cpp
    SerializeToSql.cpp
    SessionInfo.cpp
    SpeculativeTopN.cpp
    StreamingTopN.cpp
    StringDictionaryGenerations.cpp
    TableGenerations.cpp
    TargetExprBuilder.cpp
    Utils/DiamondCodegen.cpp
    StringDictionaryTranslationMgr.cpp
    StringFunctions.cpp
    StringOpsIR.cpp
    RegexpFunctions.cpp
    Visitors/SubQueryCollector.cpp
    WindowContext.cpp
    WindowExpressionRewrite.cpp
    WindowFunctionIR.cpp
    QueryPlanDagCache.cpp
    QueryPlanDagExtractor.cpp
    DataRecycler/HashtableRecycler.cpp
    DataRecycler/HashingSchemeRecycler.cpp
    Visitors/QueryPlanDagChecker.cpp
    WorkUnitBuilder.cpp

    Codec.h
    Execute.h
    NvidiaKernel.h
    QueryTemplateGenerator.h)

set(query_engine_cuda_source_files
    TopKSort.cu
    InPlaceSortImpl.cu
    ResultSetSortImpl.cu
    GpuInitGroups.cu
    JoinHashTable/Runtime/HashJoinRuntimeGpu.cu)


set(group_by_hash_test_files
        GroupByHashTest.cpp
        MurmurHash.cpp
        DynamicWatchdog.cpp
        RuntimeFunctions.cpp
        )

if (ENABLE_DECODERS_BOUNDS_CHECKING)
    list(APPEND MAPD_DEFINITIONS "-DWITH_DECODERS_BOUNDS_CHECKING")
endif ()

if (NOT PREFER_STATIC_LIBS)
    list(APPEND MAPD_DEFINITIONS "-DBOOST_LOG_DYN_LINK")
endif ()

if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
    list(APPEND MAPD_DEFINITIONS "-DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS")
endif ()

find_program(llvm_clangpp_cmd REQUIRED NAMES clang++ PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(llvm_as_cmd REQUIRED NAMES llvm-as PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(llvm_link_cmd REQUIRED NAMES llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
message(STATUS "llvm_as_cmd=${llvm_as_cmd}")
message(STATUS "llvm_clangpp_cmd=${llvm_clangpp_cmd}")
message(STATUS "llvm_link_cmd=${llvm_link_cmd}")

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


if (ENABLE_JIT_DEBUG AND NOT ENABLE_CUDA)
    set(RT_OPT_FLAGS -O0 -g)
else ()
    set(RT_OPT_FLAGS -O3)
endif ()

if(MSVC)
  message(STATUS "MSVC BUILD removing RT_OPT_FLAGS")
  unset(RT_OPT_FLAGS)
endif()

add_library(QueryEngine ${query_engine_source_files} $<$<BOOL:ENABLE_CUDA>:${query_engine_cuda_source_files}>)
add_dependencies(QueryEngine QueryEngineFunctionsTargets)

set(cpu_runtime_function_sources RuntimeFunctions.cpp DateAdd.cpp DateTruncate.cpp)
set(intel_gpu_runtime_function_sources l0_mapd_rt.cpp DateAdd.cpp DateTruncate.cpp)


set(hdk_default_runtime_functions_module_dependencies
    ${cpu_runtime_function_sources}
    RuntimeFunctions.h
    DecodersImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Utils/ExtractFromTime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Utils/StringLike.cpp
    GroupByRuntime.cpp
    TopKRuntime.cpp)

# Adds a custom command for producing llvm bitcode modules out of a .cpp source.
# Any additional arguments are treated as clang args and bypassed as-is.
function(precompile_llvm_module SOURCE_FILE SUFFIX)
    get_filename_component(src_name ${SOURCE_FILE} NAME_WLE)
    set(result_module_name ${src_name}${SUFFIX})
    message(STATUS "Adding precompiled llvm module ${result_module_name}")
    add_custom_command(
        DEPENDS ${hdk_default_runtime_functions_module_dependencies} ${SOURCE_FILE}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${result_module_name}
        COMMAND ${llvm_clangpp_cmd}
        ARGS -std=c++17 ${RT_OPT_FLAGS} ${ARGN} -c -emit-llvm
        ${CLANG_SDK_INC} ${CLANG_CRT_INC} ${MAPD_DEFINITIONS} -DEXECUTE_INCLUDE
        -o ${CMAKE_CURRENT_BINARY_DIR}/${result_module_name}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
    )
endfunction()

# Helper functions to provide easy access to precompile_llvm_module()
# Funtion names are exported to be used by precompile_modules() to dispatch .bc generation.
set(intel_gpu_module_internal_suffix "L0_internal.bc")
function(precompile_intel_gpu_module SOURCE_FILE)
    set(module_compiler_flags -Xclang -ffake-address-space-map -DL0_RUNTIME_ENABLED)
    precompile_llvm_module(${SOURCE_FILE} ${intel_gpu_module_internal_suffix} ${module_compiler_flags})
endfunction()
set(precompile_intel_gpu_module_cmd "precompile_intel_gpu_module")

set(cpu_module_internal_suffix "_internal.bc")
function(precompile_cpu_module SOURCE_FILE)
    precompile_llvm_module(${SOURCE_FILE} ${cpu_module_internal_suffix})
endfunction()
set(precompile_cpu_module_cmd "precompile_cpu_module")

# Entry point for generating all .bc files required for a runtime module
# Adds custom rules for each .cpp file provided
# Returns a list of .bc targets
function(precompile_modules OUT_LIST_NAME MODULE_SUFFIX COMPILE_COMMAND SOURCES_LIST)
    foreach(module ${SOURCES_LIST})
        cmake_language(CALL ${COMPILE_COMMAND} ${module})
        get_filename_component(mod_name ${module} NAME_WLE)
        list(APPEND precompiled_module_list "${mod_name}${MODULE_SUFFIX}")
    endforeach()
    set(${OUT_LIST_NAME} ${precompiled_module_list} PARENT_SCOPE)
endfunction()

# Links final runtime .bc module with a name provided by MOD_NAME
function(link_runtime_module MOD_NAME PRECOMPILED_MODULE_LIST)
    add_custom_command(
        DEPENDS ${PRECOMPILED_MODULE_LIST}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MOD_NAME}
        COMMAND ${llvm_link_cmd} ARGS -f ${PRECOMPILED_MODULE_LIST} -o ${CMAKE_CURRENT_BINARY_DIR}/${MOD_NAME}
    )
endfunction()

set(cpu_module_name RuntimeFunctions.bc)
set(intel_gpu_module_name RuntimeFunctionsL0.bc)

precompile_modules("intel_gpu_precompiled_module_list" ${intel_gpu_module_internal_suffix} ${precompile_intel_gpu_module_cmd} "${intel_gpu_runtime_function_sources}")
precompile_modules("cpu_precompiled_module_list" ${cpu_module_internal_suffix} ${precompile_cpu_module_cmd} "${cpu_runtime_function_sources}")

link_runtime_module(${intel_gpu_module_name} "${intel_gpu_precompiled_module_list}")
link_runtime_module(${cpu_module_name} "${cpu_precompiled_module_list}")

# SPIRV helper functions & intrinsics
set(spirv_helper_functions_module genx.bc)
add_custom_command(
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Compiler/genx.ll
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/genx.bc
    COMMAND ${llvm_as_cmd} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/Compiler/genx.ll -o ${CMAKE_CURRENT_BINARY_DIR}/${spirv_helper_functions_module}
)

if(ENABLE_L0)
    add_custom_target(IntelGPURuntimeModule DEPENDS ${intel_gpu_module_name} ${spirv_helper_functions_module})
    add_dependencies(QueryEngine IntelGPURuntimeModule)
endif()

find_package(PythonInterp REQUIRED)

add_custom_command(
        DEPENDS ExtensionFunctions.hpp ExtensionFunctionsArray.hpp ExtensionFunctionsTesting.hpp
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw
        COMMAND ${llvm_clangpp_cmd}
        ARGS -DNO_BOOST -std=c++17 -fsyntax-only -Xclang -ast-dump -fno-diagnostics-color -Wno-return-type-c-linkage
        -I ${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw)

add_custom_command(
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/parse_ast.py ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)

add_custom_target(QueryEngineFunctionsTargets
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
        ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
        )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/OmniSciTypes.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
set(query_engine_install_artefacts
    ${CMAKE_CURRENT_BINARY_DIR}/OmniSciTypes.h
    ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
    ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)
if(ENABLE_L0)
    list(APPEND query_engine_install_artefacts
        ${CMAKE_CURRENT_BINARY_DIR}/${intel_gpu_module_name}
        ${CMAKE_CURRENT_BINARY_DIR}/${spirv_helper_functions_module})
endif()
install(FILES ${query_engine_install_artefacts} DESTINATION QueryEngine COMPONENT "QE")

target_precompile_headers(QueryEngine PRIVATE
        [["QueryEngine/Execute.h"]]
        )

set(QUERY_ENGINE_LIBS
        DataMgr
        OSDependent
        Analyzer
        StringDictionary
        Utils
        Logger
        ResultSet
        ResultSetRegistry
        Shared
        SchemaMgr
        SqliteConnector
        SQLite::SQLite3
        ${Arrow_LIBRARIES}
        )

list(APPEND QUERY_ENGINE_LIBS ${llvm_libs} ${ZLIB_LIBRARIES})


if(ENABLE_COST_MODEL)
    add_subdirectory(CostModel)
    list(APPEND QUERY_ENGINE_LIBS CostModel)
endif()

target_link_libraries(QueryEngine ${QUERY_ENGINE_LIBS})

string(REPLACE " " ";" CUDA_FLAGS_LIST "$ENV{CUDAFLAGS}")
add_custom_command(
        DEPENDS cuda_mapd_rt.cu 
            JoinHashTable/Runtime/JoinHashTableQueryRuntime.cpp 
            GpuInitGroupsImpl.cpp 
            GpuInitGroups.cu 
            GroupByRuntime.cpp 
            TopKRuntime.cpp 
            StringFunctions.cpp
            RegexpFunctions.cpp
            ../Utils/ChunkIter.cpp
            ../Utils/StringLike.cpp
            ../Utils/Regexp.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctionsTesting.hpp
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
        COMMAND nvcc
        ${CUDA_FLAGS_LIST}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/../
        -I ${CMAKE_CURRENT_BINARY_DIR}
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        -Xcompiler -Wno-return-type-c-linkage --expt-relaxed-constexpr
        ${MAPD_DEFINITIONS}
        -fatbin
        -rdc=true
        ${CUDA_COMPILATION_ARCH}
        -std=c++17
        ${NVCC_BUILD_TYPE_ARGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
)

if(ENABLE_CUDA)
  add_custom_target(QueryEngineCudaTargets
    ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
    )
  string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)
  if(CMAKE_BUILD_TYPE_UPPERCASE MATCHES DEBUG)
    if (ENABLE_CUDA_KERNEL_DEBUG)
        target_compile_options(QueryEngine PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --device-debug>)
    endif()
  endif()
endif()

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test gtest Logger Shared ${Boost_LIBRARIES} ${PROFILER_LIBS})
