#!/bin/sh

./runtests.sh $* --with-coverage --cover-html --cover-package=bglibpy
firefox test/cover/index.html &
