#!/bin/sh

set -e

SRC_DIR=$1
INSTALL_DIR=$2
PYTHON_BIN=$3

PYTHON_LIB=${PYTHON_BIN}/../lib
echo ${PYTHON_LIB}
if [ ! -e ${INSTALL_DIR}/.install_finished ]
then
    echo 'Neuron was not fully installed in previous build, installing ...'
    mkdir -p ${SRC_DIR}
    cd ${SRC_DIR}
    if [ ! -d nrn ]
    then
        echo "Downloading NEURON from github ..."
        git clone https://github.com/neuronsimulator/nrn.git --depth 15 >download.log 2>&1
    else
        echo "Neuron is already downloaded"
    fi
    cd nrn
    echo "Building NEURON ..."
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_TESTS=OFF -DNRN_ENABLE_RX3D=OFF -DNRN_ENABLE_MPI=OFF >build.log 2>&1
    echo "Installing NEURON ..."
    make -j10 install >makeinstall.log 2>&1

    export PATH="${INSTALL_DIR}/x86_64/bin":${PATH}
    export PYTHONPATH="${INSTALL_DIR}/lib/python":${PYTHONPATH}

    echo "Testing NEURON import ...."
    python -c 'import neuron' >testimport.log 2>&1

    touch -f ${INSTALL_DIR}/.install_finished
    echo "NEURON successfully installed"
else
    echo 'Neuron was successfully installed in previous build, not rebuilding'
fi
