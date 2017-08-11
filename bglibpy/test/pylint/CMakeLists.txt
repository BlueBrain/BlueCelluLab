configure_file(run_pylint.sh.in run_pylint.sh @ONLY) 

add_custom_target(pylint  ${CMAKE_CURRENT_BINARY_DIR}/run_pylint.sh)
