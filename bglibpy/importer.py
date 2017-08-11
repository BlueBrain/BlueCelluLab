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

if bluepy_major_version > 0 or bluepy_minor_version > 9:
    raise Exception(
        'Version of BluePy used: %s, '
        'BGLibPy doesnt support support BluePy >= 0.10.0. '
        'After this version certain functionality is deprecated, and no'
        ' stable replacement is in place until BluePy is open sourced. '
        'Please install BluePy == 0.9.0' %
        bluepy_version)


def print_header():
    """Print BGLibPy header to stdout"""
    print "Imported neuron from %s" % neuron.__file__
    print 'HOC_LIBRARY_PATH: ', os.environ["HOC_LIBRARY_PATH"]
    print "Imported bluepy from %s" % bluepy.__file__
