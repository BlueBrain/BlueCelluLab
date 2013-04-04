# Add target doc_html_[cpp|python]: run sphinx in [cpp|python] sphinx
# projects


if (SPHINX_FOUND)
  message(STATUS "Enabling generation of html-documentation")

  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
      FILES_MATCHING
      PATTERN "source/*.rst"
      PATTERN "source/*.conf"
      PATTERN "source/*.html"
      PATTERN "source/*.png"
      PATTERN "source/*.css*"
      PATTERN "source/_templates/autosummary/*"
      PATTERN "sphinxext/numpydoc/*"
      PATTERN "source/_static/*.png"
      PATTERN "Makefile"
      PATTERN "CMakeFiles" EXCLUDE)

  configure_file(source/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/doc/source/conf.py)

  set( ENV{PYTHONPATH} ENV{PYTHONPATH}:/home/vangeit/local/bglibpy/lib64/python2.6/site-packages/ )
  add_custom_target(doc_html_python
      COMMAND ${CMAKE_MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")

  add_custom_target(doc_html)
  add_dependencies(doc_html doc_html_python)

  add_custom_target(doc)
  add_dependencies(doc doc_html)

  add_custom_target(doc_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/doc/build/html"
      "${CMAKE_INSTALL_PREFIX}/doc/html")
  add_dependencies(doc_install doc)

endif()

