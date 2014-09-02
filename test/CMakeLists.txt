option(ENABLE_COVERAGE "Enable coverage report." ON)
option(ENABLE_XUNITS "Enable XUnits reports." ON)

add_subdirectory(pylint)
add_subdirectory(pep8)

if (IS_DIRECTORY "/bgscratch")
    set(NOSEATTRIBUTES "")
else()
    set(NOSEATTRIBUTES "-A \"not bgscratch\"")
endif ()

configure_file(runtests.sh.in runtests.sh @ONLY)

set(EXTRA_NRMECH_PATH "${CMAKE_CURRENT_BINARY_DIR}/test_mod/.libs/libnrnmech.so.0.0.0")

configure_file(test_load.py.in test_load.py @ONLY)


add_custom_target(extra_libnrnmech                                        
                    WORKING_DIRECTORY test_mod                                        
                    COMMAND ${NRN_BINARIES}/nrnivmodl)


add_test(NAME NoseTests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/runtests.sh)
