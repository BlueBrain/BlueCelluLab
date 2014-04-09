# Add target doc_html_[cpp|python]: run sphinx in [cpp|python] sphinx
# projects


if (SPHINX_FOUND)
  message(STATUS "Enabling generation of html-documentation")

  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" 
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
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

  configure_file(source/conf.py.in 
      ${CMAKE_CURRENT_BINARY_DIR}/doc/source/conf.py)

  set(ENV{PYTHONPATH} 
      ENV{PYTHONPATH}:/home/vangeit/local/bglibpy/lib64/python2.6/site-packages/)
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

  if ((GIT_EXECUTABLE) AND (NOT BGLIBPY_VERSION MATCHES ".*dirty.*"))
      message("Git found, enabling doc_upload target")

      set(NEWDOC_DIRECTORY 
          "${CMAKE_CURRENT_BINARY_DIR}/Documentation/BGLibPy-${BGLIBPY_MAINVERSION}")

      configure_file(upload_doc.sh.in 
          ${CMAKE_CURRENT_BINARY_DIR}/upload_doc.sh
          @ONLY)
 
      file(COPY ${CMAKE_CURRENT_BINARY_DIR}/upload_doc.sh
          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
          FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
          GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

      add_custom_target(doc_upload 
          COMMAND ./upload_doc.sh
          DEPENDS doc)
  elseif (BGLIBPY_VERSION MATCHES ".*dirty.*")
      message("WARNING: BGLIBPY directory is dirty, disabling doc_upload target, commit changes first")
  elseif (GIT_EXECUTABLE)
      message("WARNING: Git not found, disabling doc_upload target")
  endif()

endif()
