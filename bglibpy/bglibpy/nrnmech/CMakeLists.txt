###
# LIBNRN
###

if (APPLE)
  execute_process(COMMAND uname -a COMMAND rev COMMAND cut -d " " -f1 COMMAND rev OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
else ()
  execute_process(COMMAND arch OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

if (BGP)
   set(NRNARCH powerpc64)
else (BGP)
   set(NRNARCH ${ARCH})
endif (BGP)

if (NOT NRNPATH)
   if (BBPSANTIAGO)
     set(NRNPATH /home/vangeit/local/santiago/nrnbglibpy)
   elseif (BBPQUANTAL)
     set(NRNPATH /home/vangeit/local/quantal/nrnbglibpy/)
 elseif (BBPSAUCY)
     set(NRNPATH /home/vangeit/local/saucy/nrnbglibpy/)
   else ()
      message(FATAL_ERROR "The user should set -DNRNPATH= to a directory containing a Neuron installation compiled with the linux_nrnmech=yes option
        OR define the machine name, like with -DBBPLINSRV1=1 or -DBBPSG1=1 or -DBGP=1 or -DBBPLINSRV2=1 or -DBBPNATTY=1 or -DBBPONEIRIC=1")
   endif()
endif()

set(NRN_BINARIES ${NRNPATH}/${NRNARCH}/bin)
if (NOT EXISTS ${NRN_BINARIES}/nrnivmodl)
    set(NRN_BINARIES ${NRNPATH}/bin)
    if (NOT EXISTS ${NRN_BINARIES}/nrnivmodl)
        message(FATAL_ERROR "Couldn't find nrnivmodl in ${NRNPATH}/${NRNARCH}/bin or ${NRNPATH}/bin")
    endif ()
endif ()


exec_program(
    ${NRN_BINARIES}/nrniv
    ARGS "--version" 
    OUTPUT_VARIABLE NRN_VERSION)
message("Working with Neuron Version: ${NRN_VERSION}")
message("Neuron installation path: ${NRNPATH}")

if (NOT MODLIBPATH)
    message(FATAL_ERROR "The user should set MODLIBPATH to a directory where the BBP Modlib is installed")
elseif (NOT EXISTS ${MODLIBPATH})
    message(FATAL_ERROR "MODLIBPATH ${MODLIBPATH} doesn't exist")
endif ()

if (NOT HOCLIBPATH)
    message(FATAL_ERROR "The user should set HOCLIBPATH to a directory where the BBP Hoclib is installed")
elseif (NOT EXISTS ${HOCLIBPATH})
    message(FATAL_ERROR "HOCLIBPATH ${HOCLIBPATH} doesn't exist")
endif ()

exec_program(
    "git"
    ${HOCLIBPATH}
    ARGS "describe --tags --abbrev=4 --dirty" 
    OUTPUT_VARIABLE BG_HOCLIB_VERSION)
message("Working with BG_Hoclib Version: ${BG_HOCLIB_VERSION}")

###
# MODLIB
###

set(MODLIBBUILDPATH "${CMAKE_CURRENT_BINARY_DIR}/modlib")
file(MAKE_DIRECTORY ${MODLIBBUILDPATH})
file(GLOB MODLIBMODS "${MODLIBPATH}/*.mod")
string(REGEX REPLACE "${MODLIBPATH}/hdf5.mod" ""                                 
        MODLIBMODS "${MODLIBMODS}")                                                  
string(REGEX REPLACE "${MODLIBPATH}/hdf5new.mod" ""                              
        MODLIBMODS "${MODLIBMODS}")                                
string(REGEX REPLACE "${MODLIBPATH}/HDF5record.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/HDF5reader.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/BinReportHelper.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/BinReports.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/MemUsage.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/VirtualElectrode.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/ISynNMDAEx.mod" "" MODLIBMODS "${MODLIBMODS}")                                                 
string(REGEX REPLACE "${MODLIBPATH}/ISynAMPAEx.mod" "" MODLIBMODS "${MODLIBMODS}")                           

string(REGEX REPLACE "\\.mod" ".c" MODLIBSRCS "${MODLIBMODS}")
set(MODLIBSRCS ${MODLIBSRCS} "${MODLIBPATH}/mod_func.c")
string(REGEX REPLACE "${MODLIBPATH}/" "${MODLIBBUILDPATH}/${NRNARCH}/" MODLIBSRCS "${MODLIBSRCS}")

set(MODLIBBUILDMODS "")
foreach (MODFILE ${MODLIBMODS})
    string(REGEX REPLACE "${MODLIBPATH}/" "" MODFILESHORT ${MODFILE})
    add_custom_command(
        OUTPUT "${MODLIBBUILDPATH}/${MODFILESHORT}"
        DEPENDS ${MODFILE} 
        COMMAND cmake -E copy "${MODFILE}"
        "${MODLIBBUILDPATH}/${MODFILESHORT}")
    set(MODLIBBUILDMODS "${MODLIBBUILDPATH}/${MODFILESHORT}" ${MODLIBBUILDMODS})
endforeach ()

set(LIBNRNMECHS "${MODLIBBUILDPATH}/${NRNARCH}/.libs/libnrnmech.so" "${MODLIBBUILDPATH}/${NRNARCH}/.libs/libnrnmech.so.0.0.0" )
set(LIBNRNMECH "libnrnmech.so" )

add_custom_command(OUTPUT ${LIBNRNMECHS} 
    DEPENDS ${MODLIBMODS} ${MODLIBBUILDMODS} ${SYNUTILSINC} 
    WORKING_DIRECTORY ${MODLIBBUILDPATH} 
    COMMAND ${NRN_BINARIES}/nrnivmodl)

add_custom_target(nrnmech
    COMMAND cmake -E copy "${MODLIBBUILDPATH}/${NRNARCH}/.libs/libnrnmech.so.0.0.0" ${LIBNRNMECH} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    DEPENDS ${LIBNRNMECHS})
