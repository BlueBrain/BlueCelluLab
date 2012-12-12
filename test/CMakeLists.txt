configure_file(runtests.sh.in runtests.sh @ONLY)
add_test(NAME NoseTests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/runtests.sh)
