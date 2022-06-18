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

if(MSVC)
    # set(SETUP_LDFLAGS "-L$<TARGET_FILE_DIR:Calcite> -L$<TARGET_FILE_DIR:ArrowStorage> -L$<TARGET_FILE_DIR:QueryEngine> -L$<TARGET_FILE_DIR:DataMgr> -L$<TARGET_FILE_DIR:SchemaMgr> -L$<TARGET_FILE_DIR:Shared>")
    # set(SETUP_LDFLAGS "-L $<TARGET_FILE_DIR:Calcite>;$<TARGET_FILE_DIR:ArrowStorage>;$<TARGET_FILE_DIR:QueryEngine>;$<TARGET_FILE_DIR:DataMgr>;$<TARGET_FILE_DIR:SchemaMgr>;$<TARGET_FILE_DIR:Shared>")
    # set(SETUP_LDFLAGS "-L$<TARGET_FILE_DIR:QueryEngine>;$<TARGET_FILE_DIR:Shared>")

    # set(SETUP_LDFLAGS "/LIBPATH:$<TARGET_FILE_DIR:Calcite> /LIBPATH:$<TARGET_FILE_DIR:ArrowStorage> /LIBPATH:$<TARGET_FILE_DIR:QueryEngine> /LIBPATH:$<TARGET_FILE_DIR:DataMgr> /LIBPATH:$<TARGET_FILE_DIR:SchemaMgr> /LIBPATH:$<TARGET_FILE_DIR:Shared>")
    # set(SETUP_LDFLAGS "LIBPATH=$<TARGET_FILE_DIR:QueryEngine>")
    # set(SETUP_LDFLAGS "$env:LIBPATH+=\";$<TARGET_FILE_DIR:QueryEngine>\" ;")
    set(SETUP_LDFLAGS "\"-L$<TARGET_FILE_DIR:Calcite>\;$<TARGET_FILE_DIR:ArrowStorage>\;$<TARGET_FILE_DIR:QueryEngine>\;$<TARGET_FILE_DIR:DataMgr>\;$<TARGET_FILE_DIR:SchemaMgr>\;$<TARGET_FILE_DIR:Shared>\"")
    # TODO: add -g based on CMAKE BUILD TYPE 
    set(SETUP_FLAGS -p win-amd64 -f -I ${CMAKE_CURRENT_SOURCE_DIR})
else()
    set(SETUP_LDFLAGS "-L$<TARGET_FILE_DIR:Calcite> -L$<TARGET_FILE_DIR:ArrowStorage> -L$<TARGET_FILE_DIR:QueryEngine> -L$<TARGET_FILE_DIR:DataMgr> -L$<TARGET_FILE_DIR:SchemaMgr> -L$<TARGET_FILE_DIR:Shared>")
    set(SETUP_FLAGS -g -f -I ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(MSVC)
    file(GENERATE OUTPUT "test_$<CONFIG>.out" CONTENT "${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} ${SETUP_LDFLAGS}")
    add_custom_target(pyhdk ALL
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} ${SETUP_LDFLAGS}
        DEPENDS Calcite ${pydeps}
    )
else()
    add_custom_target(pyhdk ALL
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && LDFLAGS=${SETUP_LDFLAGS} ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS}
        DEPENDS Calcite ${pydeps}
    )
endif()

if(MSVC)
# TODO
else()
    add_custom_target(pyhdk-install
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && LDFLAGS=${SETUP_LDFLAGS} ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS} install
        DEPENDS Calcite ${pydeps}
    )
endif()

string(REPLACE ";" " " SETUP_FLAGS_STR "${SETUP_FLAGS}")

if(MSVC)
# TODO
else()
    install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND bash -c \"LDFLAGS='${SETUP_LDFLAGS}' ${Python3_EXECUTABLE} ${SETUP_PY} build_ext ${SETUP_FLAGS_STR} install\")")
endif()