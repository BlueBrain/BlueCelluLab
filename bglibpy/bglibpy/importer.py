#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main importer of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import os

#####
# Load paths.config
#####

installdir = os.path.dirname(__file__)
pathsconfig_filename = installdir + "/paths.config"

if os.path.exists(pathsconfig_filename):
    pathsconfig_file = open(pathsconfig_filename, "r")
    pathsconfig = dict((line.strip().split("=")[0],
                        line.strip().split("=")[1])
                       for line in pathsconfig_file)
    pathsconfig_file.close()
else:
    raise Exception("Sorry, can not find the file paths.config")


#####
# Set HOC_LIBRARY_PATH
#####

hoc_library_path = pathsconfig["HOC_LIBRARY_PATH"]
if pathsconfig["HOC_LIBRARY_PATH"] == '':
    raise Exception('Empty HOC_LIBRIRAY_PATH')
else:
    for path in hoc_library_path.split(':'):
        if not os.path.exists(path):
            raise Exception(
                'Invalid HOC_LIBRARY_PATH: %s' %
                pathsconfig["HOC_LIBRARY_PATH"])
    os.environ["HOC_LIBRARY_PATH"] = hoc_library_path
print 'HOC_LIBRARY_PATH: ', os.environ["HOC_LIBRARY_PATH"]


#####
# Import Neuron
#####

try:
    import neuron
except ImportError:
    raise Exception('Unable to import neuron')

print "Imported neuron from %s" % neuron.__file__

#####
# Load Neuron mechanisms
#####

nrnmech_path = os.path.join(os.path.dirname(__file__), 'nrnmech/libnrnmech.so')
if not os.path.exists(nrnmech_path):
    raise Exception('Library with Neuron mechanisms does not exist at %s' %
                    nrnmech_path)
neuron.h.nrn_load_dll(nrnmech_path)


def load_extra_nrnmech(extra_nrnmech_path):
    """Load an extra nrnmech shared library.

    Parameters
    ----------
    extra_nrnmech_path : path to extra libnrnmech.so you want to load

    """

    neuron.h.nrn_load_dll(extra_nrnmech_path)


#####
# Load Neuron hoc files
#####

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("Cell.hoc")
neuron.h.load_file("TDistFunc.hoc")
neuron.h.load_file("SerializedSections.hoc")
neuron.h.load_file("TStim.hoc")
neuron.h.load_file("ShowProgress.hoc")
neuron.h('obfunc new_IClamp() { return new IClamp($1) }')
neuron.h('objref p')
neuron.h('p = new PythonObject()')


#####
# Import bluepy
#####

try:
    import bluepy
except ImportError:
    raise Exception('Unable to import bluepy')

print "Imported bluepy from %s" % bluepy.__file__
