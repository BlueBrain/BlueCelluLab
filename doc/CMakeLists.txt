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
      PATTERN "source/haiku-bbp/*"
      PATTERN "source/haiku-bbp/static/*"
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


  ###
  # Target doc_install
  ###

  add_custom_target(doc_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/doc/build/html"
      "${CMAKE_INSTALL_PREFIX}/doc/html")
  add_dependencies(doc_install doc)

      
  ###
  # Target doc_upload: Uploading documentation to bbp documentation server
  ###
  set(DOCUMENTATION_REPO ssh://bbpcode.epfl.ch/common/Documentation)

  find_package(Git)
  if(NOT GIT_EXECUTABLE)
      message(FATAL_ERROR "Git not found, incompatible with INSTALL_DOC option")
  endif()

  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Documentation" COMMAND ${GIT_EXECUTABLE} clone ${DOCUMENTATION_REPO})
  
  add_custom_target(git_pull_bbpdocumentation COMMAND ${GIT_EXECUTABLE} pull WORKING_DIRECTORY "Documentation" DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Documentation")
  
  set(NEWDOC_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Documentation/BGLibPy-2.0")
  add_custom_target(remove_bglibpy_bbpdocumentation
     COMMAND ${CMAKE_COMMAND} -E remove_directory
     ${NEWDOC_DIRECTORY} DEPENDS git_pull_bbpdocumentation)

  add_custom_target(copy_to_bbpdocumentation
     COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/doc/build/html"
      ${NEWDOC_DIRECTORY} DEPENDS remove_bglibpy_bbpdocumentation doc)

  add_custom_target(git_add_bbpdocumentation COMMAND ${GIT_EXECUTABLE} add ${NEWDOC_DIRECTORY} WORKING_DIRECTORY "Documentation" DEPENDS copy_to_bbpdocumentation)

  add_custom_target(cmake_bbpdocumentation
      COMMAND ${CMAKE_COMMAND} -DCMAKE_SOURCE_DIR="Documentation" WORKING_DIRECTORY "Documentation" DEPENDS git_add_bbpdocumentation)

  add_custom_target(git_addindex_bbpdocumentation COMMAND ${GIT_EXECUTABLE} add ${NEWDOC_DIRECTORY}/index.html WORKING_DIRECTORY "Documentation" DEPENDS cmake_bbpdocumentation)

  add_custom_target(git_commit_bbpdocumentation COMMAND ${GIT_EXECUTABLE} commit -m "Documentation for BGLibPy-${BGLIBPY_VERSION}" WORKING_DIRECTORY "Documentation" DEPENDS git_addindex_bbpdocumentation)

  # git push origin master ${DOC_PUSH_OPTION}

  add_custom_target(doc_upload DEPENDS git_commit_bbpdocumentation)
 
endif()

