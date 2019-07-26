#!/bin/sh

set -e

INSTALL_DIR=$1

if [ ! -e ${INSTALL_DIR}/.install_finished ]
then
    echo 'Neurodamus was not fully installed in previous build, installing ...'
    mkdir -p ${INSTALL_DIR}
    cd ${INSTALL_DIR}

    echo "Downloading neurodamus core ..." 
    rm -rf neurodamus-core
    git clone ssh://vangeit@bbpcode.epfl.ch/sim/neurodamus-core
    
    echo "Downloading neocortex channels ..."
    rm -rf neocortex
    git clone --depth 1 --recursive ssh://vangeit@bbpcode.epfl.ch/sim/models/neocortex
 
    echo "Building mod files"
    nrnivmodl neocortex/mod/v6 >nrnivmodl.log 2>&1

    touch -f ${INSTALL_DIR}/.install_finished
    echo "Neurodamus successfully installed"
else
    echo 'Neurodamus was successfully installed in previous build, not rebuilding'
fi
