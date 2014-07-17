set(NRNMECH_PATH "nrnmech/libnrnmech.so") 

add_subdirectory(bglibpy)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY) 

add_custom_command(OUTPUT ${PYTHON_PACKAGE_FILE}
    COMMAND ${CMAKE_COMMAND} -E tar czf ${PYTHON_PACKAGE_FILE} bglibpy/setup.py bglibpy/bglibpy/*.py bglibpy/bglibpy/${NRNMECH_PATH} bglibpy/bglibpy/paths.config
    DEPENDS expand_PYTHON_PACKAGE_TMP_FILE ${PYTHON_PACKAGE_CONTENT_DIR} nrnmech
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

option(ENABLE_PIP "Enable the use of pip to install the python package" ON)

if (ENABLE_PIP)
    add_custom_target(pip_package ALL DEPENDS ${PYTHON_PACKAGE_FILE})
    install(CODE
        "execute_process(COMMAND pip install ${PYTHON_PACKAGE_FILE} --upgrade)"
    )
else ()
    install(CODE
        "execute_process(
                COMMAND python setup.py install --prefix=${CMAKE_INSTALL_PREFIX}
                WORKING_DIRECTORY ${PYTHON_PACKAGE_CONTENT_DIR})"
    )
endif ()
