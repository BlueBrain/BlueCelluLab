add_subdirectory(nrnmech)

set(HOC_LIBRARY_PATH "${HOCLIBPATH}:${NRNPATH}/share/nrn/lib/hoc/")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/versions.py.in ${CMAKE_CURRENT_BINARY_DIR}/versions.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths.config.in ${CMAKE_CURRENT_BINARY_DIR}/paths.config)

set(PYTHON_FILES __init__.py tools.py ssim.py cell.py simulation.py psection.py psegment.py plotwindow.py dendrogram.py importer.py connection.py synapse.py) 

set(PYTHON_PACKAGE_TMP_FILE ${CMAKE_BINARY_DIR}/pythonfiles_tmp.tar.gz)
set(PYTHON_PACKAGE_CONTENT_DIR  ${CMAKE_BINARY_DIR}/bglibpy)

add_custom_command(OUTPUT ${PYTHON_PACKAGE_TMP_FILE}
    COMMAND ${CMAKE_COMMAND} -E tar czf ${PYTHON_PACKAGE_TMP_FILE} *.py
    DEPENDS ${PYTHON_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_custom_target(expand_PYTHON_PACKAGE_TMP_FILE
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${PYTHON_PACKAGE_TMP_FILE}
    DEPENDS ${PYTHON_PACKAGE_TMP_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
