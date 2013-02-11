configure_file(runtests.sh.in runtests.sh @ONLY)
configure_file(test_load.py.in test_load.py @ONLY)
add_test(NAME NoseTests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/runtests.sh)
