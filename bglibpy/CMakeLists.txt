set(NRNMECH_PATH "nrnmech/libnrnmech.so") 

add_subdirectory(bglibpy)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY) 

set(PYTHON_PACKAGE_FILE ${CMAKE_BINARY_DIR}/bglibpy.tar.gz)
set(PYTHON_PACKAGE_CONTENT_DIR  ${CMAKE_BINARY_DIR}/bglibpy)

# add_custom_command(OUTPUT ${PYTHON_PACKAGE_CONTENT_DIR}
#    COMMAND ${CMAKE_COMMAND} -E copy ${LIBNRNMECH} ${PYTHON_PACKAGE_CONTENT_DIR}/bglibpy
#    DEPENDS ${LIBNRNMECH}
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    )

add_custom_command(OUTPUT ${PYTHON_PACKAGE_FILE}
    COMMAND ${CMAKE_COMMAND} -E tar czf ${PYTHON_PACKAGE_FILE} bglibpy/setup.py bglibpy/bglibpy/*.py bglibpy/bglibpy/${NRNMECH_PATH} bglibpy/bglibpy/paths.config
    DEPENDS expand_PYTHON_PACKAGE_TMP_FILE ${PYTHON_PACKAGE_CONTENT_DIR} nrnmech
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

add_custom_target(python_package DEPENDS ${PYTHON_PACKAGE_FILE})
