set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in" "${SETUP_PY}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB_RECURSE PY_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.py)
file(GLOB_RECURSE PYX_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.pxd)
file(GLOB_RECURSE PXD_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyhdk/*.pyx)

set(pydeps
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${PY_SOURCES}
    ${PYX_SOURCES}
    ${PXD_SOURCES}
    Calcite)

set(SETUP_LDFLAGS "-L$<TARGET_FILE_DIR:Calcite> -L$<TARGET_FILE_DIR:ArrowStorage> -L$<TARGET_FILE_DIR:QueryEngine> -L$<TARGET_FILE_DIR:SchemaMgr> -L$<TARGET_FILE_DIR:ConfigBuilder> -L$<TARGET_FILE_DIR:Logger> -L$<TARGET_FILE_DIR:Shared> -L$<TARGET_FILE_DIR:DataMgr>")
set(SETUP_FLAGS -g -f -I ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(pyhdk ALL
    COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && LDFLAGS=${SETUP_LDFLAGS} ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS}
    DEPENDS Calcite ${pydeps}
)

add_custom_target(pyhdk-install
    COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && LDFLAGS=${SETUP_LDFLAGS} ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} install
    DEPENDS Calcite ${pydeps}
)

string(REPLACE ";" " " SETUP_FLAGS_STR "${SETUP_FLAGS}")
install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND bash -c \"LDFLAGS='${SETUP_LDFLAGS}' ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS_STR} install\")")
