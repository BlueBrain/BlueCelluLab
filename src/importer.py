#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main importer of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import sys
import os
import numpy

installdir = os.path.dirname(__file__)
pathsconfig_filename = installdir + "/paths.config"

if os.path.exists(pathsconfig_filename):
    pathsconfig_file = open(pathsconfig_filename, "r")
    pathsconfig = dict((line.strip().split("=")[0], line.strip().split("=")[1]) for line in pathsconfig_file)
    pathsconfig_file.close()
else:
    raise Exception("Sorry, can not find the file paths.config")

os.environ["HOC_LIBRARY_PATH"] = pathsconfig["HOC_LIBRARY_PATH"]
print 'HOC_LIBRARY_PATH: ', os.environ["HOC_LIBRARY_PATH"]

sys.path = [pathsconfig["NRNPYTHONPATH"]] + sys.path
import neuron
print "Imported neuron from %s" % neuron.__file__

neuron.h.nrn_load_dll(pathsconfig["NRNMECH_PATH"])
neuron.h.load_file("stdrun.hoc")

neuron.h.load_file("Cell.hoc")
neuron.h.load_file("TDistFunc.hoc")
neuron.h.load_file("SerializedSections.hoc")
neuron.h.load_file("TStim.hoc")
neuron.h('obfunc new_IClamp() { return new IClamp($1) }')

sys.path = [pathsconfig["BLUEPYPATH"]] + sys.path
import bluepy
