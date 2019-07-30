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
    git clone --depth 1 ssh://bbpcode.epfl.ch/sim/neurodamus-core
    
    echo "Downloading neocortex channels ..."
    rm -rf neocortex
    git clone --depth 1 --recursive ssh://bbpcode.epfl.ch/sim/models/neocortex
 
    # Remove the 4 lines below once these mod files are part of the channel repos
    cp neurodamus-core/mod/netstim_inhpoisson.mod neocortex/mod/v5
    cp neurodamus-core/mod/VecStim.mod neocortex/mod/v5
    
    cp neurodamus-core/mod/netstim_inhpoisson.mod neocortex/mod/v6
    cp neurodamus-core/mod/VecStim.mod neocortex/mod/v6

    # Remove this line once we have a canonical v6 sim
    cp neocortex/mod/v5/Ca_HVA.mod neocortex/mod/v6
    
    touch -f .install_finished
    echo "Neurodamus successfully installed"
else
    echo 'Neurodamus was successfully installed in previous build, not rebuilding'
fi
