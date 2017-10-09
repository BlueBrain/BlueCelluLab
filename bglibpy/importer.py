#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main importer of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import os

# pylint: disable=F0401


#####
# Load Neuron hoc files
#####

import neuron

if 'HOC_LIBRARY_PATH' not in os.environ:
    print "BGLibPy: HOC_LIBRARY_PATH not found, this is required to find " \
        "Neurodamus. Did you install neurodamus correctly ?"


if 'BGLIBPY_MOD_LIBRARY_PATH' in os.environ:
    mod_lib_path = os.environ["BGLIBPY_MOD_LIBRARY_PATH"]
    neuron.h.nrn_load_dll(mod_lib_path)
else:
    print "BGLibPY: BGLIBPY_MOD_LIBRARY_PATH not found, continuing with " \
        "pre-loading MOD files, assuming they have already been loaded"

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("Cell.hoc")
neuron.h.load_file("TDistFunc.hoc")
neuron.h.load_file("SerializedSections.hoc")
neuron.h.load_file("TStim.hoc")
neuron.h.load_file("ShowProgress.hoc")
neuron.h('obfunc new_IClamp() { return new IClamp($1) }')
neuron.h('objref p')
neuron.h('p = new PythonObject()')


import bluepy
bluepy_version = bluepy.__version__
bluepy_major_version = int(bluepy_version.split(".")[0])
bluepy_minor_version = int(bluepy_version.split(".")[1])


def print_header():
    """Print BGLibPy header to stdout"""
    print "Imported neuron from %s" % neuron.__file__
    print 'HOC_LIBRARY_PATH: ', os.environ["HOC_LIBRARY_PATH"]
    print 'BGLIBPY_MOD_LIBRARY_PATH: ', mod_lib_path
    print "Imported bluepy from %s" % bluepy.__file__

print_header()
