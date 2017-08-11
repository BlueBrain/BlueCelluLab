configure_file(run_pep8.sh.in run_pep8.sh @ONLY) 

add_custom_target(pep8 ${CMAKE_CURRENT_BINARY_DIR}/run_pep8.sh)
