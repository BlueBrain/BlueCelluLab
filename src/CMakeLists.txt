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
   else ()
      message(FATAL_ERROR "The user should set -DNRNPATH= to a directory containing a Neuron installation compiled with the linux_nrnmech=yes option
        OR define the machine name, like with -DBBPLINSRV1=1 or -DBBPSG1=1 or -DBGP=1 or -DBBPLINSRV2=1 or -DBBPNATTY=1 or -DBBPONEIRIC=1")
   endif()
endif()

if (NOT NRNPYTHONPATH)
    if (BBPSANTIAGO)
        set(NRNPYTHONPATH /home/vangeit/local/santiago/nrnbglibpy/lib64/python2.6/site-packages)
    else ()
        message(FATAL_ERROR "The user should set NRNPYTHONPATH to a directory where the Neuron python module is installed")
    endif()
endif ()

set(NRN_BINARIES ${NRNPATH}/${NRNARCH}/bin)

if (NOT MODLIBPATH)
    message(FATAL_ERROR "The user should set MODLIBPATH to a directory where the BBP Modlib is installed")
endif ()

if (NOT HOCLIBPATH)
    message(FATAL_ERROR "The user should set HOCLIBPATH to a directory where the BBP Hoclib is installed")
endif ()

if (NOT BLUEPYPATH)
    message(FATAL_ERROR "The user should set BLUEPYPATH to a directory where the BluePy python module is installed")
endif ()

###
# MODLIB
###

set(MODLIBBUILDPATH "${CMAKE_CURRENT_BINARY_DIR}/modlib")
file(MAKE_DIRECTORY ${MODLIBBUILDPATH})
file(GLOB MODLIBMODS "${MODLIBPATH}/*.mod")
string(REGEX REPLACE "${MODLIBPATH}/HDF5record.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/HDF5reader.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/BinReportHelper.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/BinReports.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/MemUsage.mod" "" MODLIBMODS "${MODLIBMODS}")
string(REGEX REPLACE "${MODLIBPATH}/VirtualElectrode.mod" "" MODLIBMODS "${MODLIBMODS}")

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

if (EXISTS "${MODLIBPATH}/synutils.inc")
    add_custom_command(
        OUTPUT "${MODLIBBUILDPATH}/synutils.inc" 
        COMMAND cmake -E copy "${MODLIBPATH}/synutils.inc"
        "${MODLIBBUILDPATH}/synutils.inc")
    set(SYNUTILSINC "${MODLIBBUILDPATH}/synutils.inc")
endif()

set(LIBNRNMECHS "${MODLIBBUILDPATH}/${ARCH}/.libs/libnrnmech.so" "${MODLIBBUILDPATH}/${ARCH}/.libs/libnrnmech.so.0.0.0" )
add_custom_target(bglipby ALL DEPENDS ${LIBNRNMECHS})
add_custom_command(OUTPUT ${LIBNRNMECHS} DEPENDS ${MODLIBMODS} ${MODLIBBUILDMODS} ${SYNUTILSINC} WORKING_DIRECTORY ${MODLIBBUILDPATH} COMMAND ${NRN_BINARIES}/nrnivmodl)

set(HOC_LIBRARY_PATH "${HOCLIBPATH}:${NRNPATH}/share/nrn/lib/hoc/")

set(NRNMECH_PATH "${DESTINATION_DIR}/libnrnmech.so")

configure_file(paths.config.in paths.config)

set(PYTHON_FILES __init__.py tools.py ssim.py cell.py simulation.py psection.py psegment.py plotwindow.py tools.py dendrogram.py importer.py)

install(FILES ${LIBNRNMECHS} DESTINATION ${DESTINATION_DIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/paths.config" DESTINATION ${DESTINATION_DIR})
install(FILES ${PYTHON_FILES} DESTINATION ${DESTINATION_DIR})
