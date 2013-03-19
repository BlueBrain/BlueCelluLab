# Add target doc_html_[cpp|python]: run sphinx in [cpp|python] sphinx
# projects


if (SPHINX_FOUND)
  message(STATUS "Enabling generation of html-documentation")

  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
      FILES_MATCHING
      PATTERN "*.rst"
      PATTERN "*.py"
      PATTERN "*.conf"
      PATTERN "*.html"
      PATTERN "*.png"
      PATTERN "*.css*"
      PATTERN "Makefile"
      PATTERN "CMakeFiles" EXCLUDE)

  add_custom_target(doc_html_python
      COMMAND ${CMAKE_MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")

  add_custom_target(doc_html)
  add_dependencies(doc_html doc_html_python)

  add_custom_target(doc)
  add_dependencies(doc doc_html)

  add_custom_target(doc_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/doc/_build/html"
      "${CMAKE_INSTALL_PREFIX}/doc/html")
  add_dependencies(doc_install doc)

endif()

