#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main importer of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import os

# pylint: disable=F0401


#####
# Load NEURON hoc files
#####


def _nrn_disable_banner():
    """Disable NEURON banner"""

    import importlib.util
    import ctypes

    neuron_spec = importlib.util.find_spec("neuron")
    nrnpy_path = neuron_spec.submodule_search_locations[0]
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


def import_neuron():
    """Import NEURON simulator"""
    _nrn_disable_banner()

    import neuron

    return neuron


def import_hoc_lib():
    """Check of hoc library path is present"""

    if 'HOC_LIBRARY_PATH' not in os.environ:
        raise Exception(
            "BGLibPy: HOC_LIBRARY_PATH not found, this is required to find "
            "Neurodamus. Did you install neurodamus correctly ?")
    else:
        return os.environ['HOC_LIBRARY_PATH']


def import_mod_lib(neuron):
    """Import mod files"""

    mod_lib_list = None
    if 'BGLIBPY_MOD_LIBRARY_PATH' in os.environ:
        mod_lib_path = os.environ["BGLIBPY_MOD_LIBRARY_PATH"]
        mod_lib_list = mod_lib_path.split(':')
        for mod_lib in mod_lib_list:
            neuron.h.nrn_load_dll(mod_lib)

    return mod_lib_list


def import_neurodamus(neuron):
    """Import neurodamus"""
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


def print_header(neuron, hoc_lib_path, mod_lib_path):
    """Print BGLibPy header to stdout"""
    print("Imported neuron from %s" % neuron.__file__)
    print('Hoc library: ', hoc_lib_path)
    print('Mod libs: ', mod_lib_path)


neuron = import_neuron()
hoc_lib_path = import_hoc_lib()
mod_lib_paths = import_mod_lib(neuron)
import_neurodamus(neuron)
# print_header(neuron, hoc_lib_path, mod_lib_paths)
