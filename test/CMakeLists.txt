option(ENABLE_COVERAGE "Enable coverage report." ON)
option(ENABLE_XUNITS "Enable XUnits reports." ON)


add_subdirectory(pylint)
add_subdirectory(pep8)

configure_file(runtests.sh.in runtests.sh @ONLY)
configure_file(test_load.py.in test_load.py @ONLY)

if (NOBGSCRATCH)
    set(NOSEATTRIBUTES "NOBGSCRATCH")
endif ()
add_test(NAME NoseTests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/runtests.sh ${NOSEATTRIBUTES})
