#!/bin/bash

set -e
set -x

git config --global http.proxy http://bbpproxy.epfl.ch:80/
git config --global https.proxy http://bbpproxy.epfl.ch:80/

tox_args='-v --recreate'

if [ "${os}" = "bb5" ]
then
	. /opt/rh/rh-python36/enable
    tox_args="${tox_args} -e py3-test-v5-v6-thal-upload_docs-devpi"
elif [ "${os}" = "cscsviz" ]
then
	. /opt/rh/python27/enable
    tox_args="${tox_args} -e py27-test-v5-v6-thal"
elif  [ "${os}" = "Ubuntu-18.04" ]
then
    tox_args="${tox_args} -e py3-test"
else
    tox_args="${tox_args} -e py27-test"
fi

cd $WORKSPACE

#########
# Virtualenv
#########

if [ ! -d "${WORKSPACE}/env" ]; then
  virtualenv ${WORKSPACE}/env --no-site-packages
fi

. ${WORKSPACE}/env/bin/activate
pip install pip --upgrade
pip install tox --upgrade   

if  [ "${os}" = "Ubuntu-16.04" ]
then
    # Newer virtualenv creates problem with tox on ubuntu-16.04 or py27
    pip install virtualenv==16.7.9
fi

#####
# Tests
#####

cd  ${WORKSPACE}/BGLibPy

tox ${tox_args}
