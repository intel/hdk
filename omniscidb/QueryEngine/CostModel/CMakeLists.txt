set(COST_MODEL_SOURCES 
        CostModel.cpp 
        DataSources/EmptyDataSource.cpp 
        ExtrapolationModels/LinearExtrapolation.cpp 
        DummyCostModel.cpp 
        DataSources/DataSource.cpp 
        Measurements.cpp)


set(COST_MODEL_LIBS Logger)

if(ENABLE_DWARF_BENCH)
  find_package(dbench REQUIRED)
  message(STATUS "Dwarf Bench found ${dbench_DIR}")
  add_definitions("-DHAVE_DWARF_BENCH")
  
  list(APPEND COST_MODEL_LIBS dbench::dbench)
  list(APPEND COST_MODEL_SOURCES DataSources/DwarfBench.cpp)
endif()

if(ENABLE_ARMADILLO)
  find_package(Armadillo REQUIRED)
  message(STATUS "Armadillo found")
  add_definitions("-DHAVE_ARMADILLO")
  
  list(APPEND COST_MODEL_LIBS ${ARMADILLO_LIBRARIES})
  list(APPEND COST_MODEL_SOURCES ExtrapolationModels/LinearRegression.cpp)
endif()

add_library(CostModel STATIC ${COST_MODEL_SOURCES})
target_link_libraries(CostModel PRIVATE ${COST_MODEL_LIBS})
target_compile_options(CostModel PRIVATE -fPIC)
