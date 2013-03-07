# @author  vangeit
# @date    2012-11-08
# @remarks Copyright (c) BBP/EPFL 2005-2012; All rights reserved. Do not distribute without further notice.

cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# Set the name of your project here
project("BGLibPy" NONE)

# Set the version number of your project here (format is MAJOR.MINOR.PATCHLEVEL - e.g. 1.0.0)
set(VERSION_MAJOR "0")
set(VERSION_MINOR "0")
set(VERSION_PATCH "1")

set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})
if(VERSION_MINOR VERSION_LESS 10)
  set(VERSION_ABI ${VERSION_MAJOR}0${VERSION_MINOR})
else()
  set(VERSION_ABI ${VERSION_MAJOR}${VERSION_MINOR})
endif()

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(True, prefix='${CMAKE_INSTALL_PREFIX}');"
    OUTPUT_VARIABLE PYTHON_PACKAGE_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(DESTINATION_DIR "${PYTHON_PACKAGE_INSTALL_DIR}/bglibpy")

add_subdirectory(src)

enable_testing()
add_subdirectory(test)
