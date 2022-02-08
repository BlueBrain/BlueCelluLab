#!/bin/sh

set -e

NEOCORTEX_DIR=$1

cd ${NEOCORTEX_DIR}

echo ${NEOCORTEX_DIR}

echo "Fetching common mod files"
./fetch_common.bash
cd -
