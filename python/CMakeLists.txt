set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

get_property(CDEFS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
get_property(IDIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

# On windows setup.py for some reason is unable to work with
# cmake-style path with forward slashes. It requires native path
# with back slashes in definition of package_dir variable.
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" NATIVE_CMAKE_CURRENT_SOURCE_DIR)
string(REPLACE "\\" "\\\\" NATIVE_CMAKE_CURRENT_SOURCE_DIR ${NATIVE_CMAKE_CURRENT_SOURCE_DIR})
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
set(SETUP_LD_PATH_LIST $<TARGET_FILE_DIR:Calcite>$<SEMICOLON>$<TARGET_FILE_DIR:IR>$<SEMICOLON>$<TARGET_FILE_DIR:ArrowStorage>$<SEMICOLON>$<TARGET_FILE_DIR:QueryBuilder>$<SEMICOLON>$<TARGET_FILE_DIR:QueryEngine>$<SEMICOLON>$<TARGET_FILE_DIR:SchemaMgr>$<SEMICOLON>$<TARGET_FILE_DIR:ConfigBuilder>$<SEMICOLON>$<TARGET_FILE_DIR:Logger>$<SEMICOLON>$<TARGET_FILE_DIR:Shared>$<SEMICOLON>$<TARGET_FILE_DIR:DataMgr>$<SEMICOLON>$<TARGET_FILE_DIR:ResultSet>$<SEMICOLON>$<TARGET_FILE_DIR:ResultSetRegistry>$<SEMICOLON>$<TARGET_FILE_DIR:${Boost_LOG_LIBRARY}>)

if (WIN32)
  # Convert all paths from library list to paths with forward
  # slashes. This is required because on windows paths with back slashes
  # work incorrectly in cmake_install.cmake since cmake thinks that
  # those are quoted characters.
  cmake_path(CONVERT ${SETUP_LD_PATH_LIST} TO_CMAKE_PATH_LIST SETUP_LDFLAGS)
else()
  # Convert semicolons from the list above into colons path separator
  # on Linux.
  set(SETUP_LDFLAGS "$<SHELL_PATH:${SETUP_LD_PATH_LIST}>")
endif()
set(SETUP_FLAGS -f -I ${CMAKE_CURRENT_SOURCE_DIR} -L ${SETUP_LDFLAGS})

if (NOT WIN32 AND (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    # On windows enabling debug in python requires python debug library which is absent in conda
    # so python with debug info is built only on linux.
    list(APPEND SETUP_FLAGS -g)
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

# On windows we cannot use variable ${SETUP_FLAGS} in the line below
# because it has all delimiting spaces converted to semicolons while
# we need spaces between arguments and semicolons in library
# list. Therefore repeat all arguments from ${SETUP_FLAGS} here. Also
# quotes are required around ${SETUP_LDFLAGS} because otherwise cmake
# converts semicolons to spaces when executing
# cmake_install.cmake. This stuff is very fragile, touch with care!
install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext -f -I ${CMAKE_CURRENT_SOURCE_DIR} -L \"${SETUP_LDFLAGS}\" install)")
