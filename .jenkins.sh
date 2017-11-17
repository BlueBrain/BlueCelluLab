#!/bin/bash

set -e
set -x

git config --global http.proxy http://bbpproxy.epfl.ch:80/
git config --global https.proxy http://bbpproxy.epfl.ch:80/

tox_args='--recreate -e py27-test'

if [ "${os}" = "cscsviz" ]
then
	. /opt/rh/python27/enable
    tox_args="${tox_args}-v5-proj64"
elif [ "${os}" = "Ubuntu-16.04" ]
then
	tox_args="${tox_args}-upload_docs-devpi"
fi

which python
python --version

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

#####
# Tests
#####

cd  ${WORKSPACE}/BGLibPy

tox ${tox_args}