set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in" "${SETUP_PY}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pyhdk/version.py.in" "${CMAKE_CURRENT_SOURCE_DIR}/pyhdk/version.py")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB_RECURSE PY_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.py)
file(GLOB_RECURSE PYX_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.pxd)
file(GLOB_RECURSE PXD_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.pyx)

set(pydeps
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/pyhdk/version.py.in
    ${PY_SOURCES}
    ${PYX_SOURCES}
    ${PXD_SOURCES}
    Calcite)

# $<SEMICOLON> is used here as a list separator and it is converted to colon ":" on Linux in $<SHELL_PATH>
# expression.
set(SETUP_LDFLAGS $<TARGET_FILE_DIR:Calcite>$<SEMICOLON>$<TARGET_FILE_DIR:IR>$<SEMICOLON>$<TARGET_FILE_DIR:ArrowStorage>$<SEMICOLON>$<TARGET_FILE_DIR:QueryBuilder>$<SEMICOLON>$<TARGET_FILE_DIR:QueryEngine>$<SEMICOLON>$<TARGET_FILE_DIR:SchemaMgr>$<SEMICOLON>$<TARGET_FILE_DIR:ConfigBuilder>$<SEMICOLON>$<TARGET_FILE_DIR:Logger>$<SEMICOLON>$<TARGET_FILE_DIR:Shared>$<SEMICOLON>$<TARGET_FILE_DIR:DataMgr>$<SEMICOLON>$<TARGET_FILE_DIR:ResultSet>$<SEMICOLON>$<TARGET_FILE_DIR:ResultSetRegistry>$<SEMICOLON>$<TARGET_FILE_DIR:${Boost_LOG_LIBRARY}>)

set(SETUP_FLAGS -f -I ${CMAKE_CURRENT_SOURCE_DIR} -L "$<SHELL_PATH:${SETUP_LDFLAGS}>")

if (NOT WIN32 AND (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    # On windows enabling debug in python requires python debug library which is absent in conda
    # so python with debug info is built only on linux.
    list(APPEND SETUP_FLAGS -g)
endif()

if(ENABLE_FOLLY)
   list(APPEND SETUP_FLAGS "-DHAVE_FOLLY") 
endif()

add_custom_target(pyhdk ALL
    COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS}
    DEPENDS Calcite ${pydeps}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(pyhdk-install
    COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} install
    DEPENDS Calcite ${pydeps}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} install)")
