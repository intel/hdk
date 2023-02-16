add_library(OSDependent ${OSDEPENDENT_SOURCE_FILES})
set_target_properties(OSDependent PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..)

target_link_libraries(OSDependent Logger)
