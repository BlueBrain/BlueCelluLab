# @author  vangeit
# @date    2012-11-08
# @remarks Copyright (c) BBP/EPFL 2005-2012; All rights reserved. Do not distribute without further notice.

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

site_name(HOSTNAME)
message("Running on hostname: " ${HOSTNAME})

# Set the name of your project here
project("BGLibPy" NONE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

###
# VERSION
###

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --tags --abbrev=4 --dirty" 
    OUTPUT_VARIABLE BGLIBPY_VERSION)
message("Working with BGLibPy Version: ${BGLIBPY_VERSION}")

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(True, prefix='${CMAKE_INSTALL_PREFIX}');"
    OUTPUT_VARIABLE PYTHON_PACKAGE_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(DESTINATION_DIR "${PYTHON_PACKAGE_INSTALL_DIR}/bglibpy")

add_subdirectory(src)

if (MODULEPATH)
    add_subdirectory(modules)
endif()

enable_testing()
add_subdirectory(test)

option(ENABLE_DOC "Enable generation of documentation." OFF)
if (ENABLE_DOC)
  find_package("Sphinx")
  if (NOT SPHINX_FOUND)
    message(STATUS "Disabling generation of documentation because Sphinx is missing.")
  else()
    add_subdirectory(doc)
  endif()
endif()
