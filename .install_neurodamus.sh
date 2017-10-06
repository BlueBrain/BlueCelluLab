#!/bin/sh

set -e

INSTALL_DIR=$1

if [ ! -e ${INSTALL_DIR}/.install_finished ]
then
    echo 'Neurodamus was not fully installed in previous build, installing ...'
    mkdir -p ${INSTALL_DIR}
    cd ${INSTALL_DIR}
    if [ ! -d ${INSTALL_DIR}/bbp ]
    then
        echo "Downloading neurodamus ..."
        git clone ssh://bbpcode.epfl.ch/sim/neurodamus/bbp.git
    else
        echo "Neurodamus already downloaded"
    fi

    cd bbp
    rm -rf lib/modlib/Bin*.mod
    rm -rf lib/modlib/HDF*.mod
    rm -rf lib/modlib/hdf*.mod
    rm -rf lib/modlib/MemUsage*.mod

    which nrnivmodl
    nrnivmodl lib/modlib

    touch -f ${INSTALL_DIR}/.install_finished
    echo "Neurodamus successfully installed"
else
    echo 'Neurodamus was successfully installed in previous build, not rebuilding'
fi
