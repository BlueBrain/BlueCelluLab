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


def _nrn_disable_banner():
    """Disable Neuron banner"""

    import imp
    import ctypes

    nrnpy_path = os.path.join(imp.find_module('neuron')[1])
    import glob
    hoc_so_list = \
        glob.glob(os.path.join(nrnpy_path, 'hoc*.so'))

    if len(hoc_so_list) != 1:
        raise Exception(
            'hoc shared library not found in %s' %
            nrnpy_path)

    hoc_so = hoc_so_list[0]
    nrndll = ctypes.cdll[hoc_so]
    ctypes.c_int.in_dll(nrndll, 'nrn_nobanner_').value = 1


_nrn_disable_banner()

import neuron

if 'HOC_LIBRARY_PATH' not in os.environ:
    raise Exception(
        "BGLibPy: HOC_LIBRARY_PATH not found, this is required to find "
        "Neurodamus. Did you install neurodamus correctly ?")


if 'BGLIBPY_MOD_LIBRARY_PATH' in os.environ:
    mod_lib_path = os.environ["BGLIBPY_MOD_LIBRARY_PATH"]
    neuron.h.nrn_load_dll(mod_lib_path)
# else:
#    print "WARNING: BGLIBPY_MOD_LIBRARY_PATH not found, continuing without " \
#        "pre-loading MOD files, assuming they have already been loaded"

neuron.h('objref simConfig')

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("defvar.hoc")
neuron.h.default_var("simulator", "NEURON")
neuron.h.load_file("Cell.hoc")
neuron.h.load_file("TDistFunc.hoc")
neuron.h.load_file("SerializedSections.hoc")
neuron.h.load_file("TStim.hoc")
neuron.h.load_file("ShowProgress.hoc")
neuron.h.load_file("SimSettings.hoc")
neuron.h.load_file("RNGSettings.hoc")

neuron.h('obfunc new_IClamp() { return new IClamp($1) }')
neuron.h('objref p')
neuron.h('p = new PythonObject()')

neuron.h('simConfig = new SimSettings()')

import bluepy
bluepy_version = bluepy.__version__
bluepy_major_version = int(bluepy_version.split(".")[0])
bluepy_minor_version = int(bluepy_version.split(".")[1])


def print_header():
    """Print BGLibPy header to stdout"""
    print("Imported neuron from %s" % neuron.__file__)
    print('HOC_LIBRARY_PATH: ', os.environ["HOC_LIBRARY_PATH"])
    print('BGLIBPY_MOD_LIBRARY_PATH: ', mod_lib_path)
    print("Imported bluepy from %s" % bluepy.__file__)
