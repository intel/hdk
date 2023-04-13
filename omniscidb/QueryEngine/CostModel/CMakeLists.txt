set(COST_MODEL_SOURCES 
        CostModel.cpp 
        DataSources/EmptyDataSource.cpp 
        ExtrapolationModels/LinearExtrapolation.cpp 
        ExtrapolationModels/LinearRegression.cpp 
        DummyCostModel.cpp 
        DataSources/DataSource.cpp 
        Measurements.cpp 
        DataSources/DwarfBench.cpp)

set(COST_MODEL_LIBS dbench::dbench ${ARMADILLO_LIBRARIES} Logger)

add_library(CostModel STATIC ${COST_MODEL_SOURCES})
target_link_libraries(CostModel PRIVATE ${COST_MODEL_LIBS})
