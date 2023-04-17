set(COST_MODEL_SOURCES 
        CostModel.cpp 
        DataSources/EmptyDataSource.cpp 
        ExtrapolationModels/LinearExtrapolation.cpp 
        ExtrapolationModels/LinearRegression.cpp 
        DummyCostModel.cpp 
        DataSources/DataSource.cpp 
        Measurements.cpp)

find_package(Armadillo REQUIRED)

set(COST_MODEL_LIBS ${ARMADILLO_LIBRARIES} Logger)

if(ENABLE_DWARF_BENCH)
  find_package(dbench REQUIRED)
  message(STATUS "Dwarf Bench found ${dbench_DIR}")
  add_definitions("-DENABLE_DWARF_BENCH")
  
  list(APPEND COST_MODEL_LIBS dbench::dbench)
  list(APPEND COST_MODEL_SOURCES DataSources/DwarfBench.cpp)
endif()

add_library(CostModel STATIC ${COST_MODEL_SOURCES})
target_link_libraries(CostModel PRIVATE ${COST_MODEL_LIBS})
