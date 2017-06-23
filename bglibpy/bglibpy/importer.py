#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main importer of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import sys
import os

# pylint: disable=F0401

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
    raise Exception('Empty HOC_LIBRARY_PATH')
else:
    paths = []
    if "HOC_LIBRARY_PATH" in os.environ:
        paths = os.environ["HOC_LIBRARY_PATH"].split(':')
    for path in hoc_library_path.split(':'):
        if not os.path.exists(path):
            print 'Path %s in HOC_LIBRARY_PATH doesn\'t exist, ignoring' % path
        else:
            paths.insert(0, path)

    os.environ["HOC_LIBRARY_PATH"] = ':'.join(paths)


#####
# Import Neuron
#####

nrn_python_path = pathsconfig["NRNPYTHONPATH"]
if nrn_python_path == '':
    try:
        import neuron  # pylint: disable=C0413
    except ImportError:
        print 'Unable to import neuron from default python path,' \
            'and NRNPYTHONPATH is not set either'
        raise
else:
    if not os.path.exists(nrn_python_path):
        raise Exception('Inexistent NRNPYTHONPATH=%s' % nrn_python_path)
    sys.path.insert(0, nrn_python_path)
    try:
        import neuron  # pylint: disable=C0413
    except ImportError:
        print 'Unable to import neuron from NRNPYTHONPATH=%s' % nrn_python_path
        raise


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

    if os.path.exists(extra_nrnmech_path):
        neuron.h.nrn_load_dll(extra_nrnmech_path)
    else:
        raise Exception("Library given to load_extra_nrnmech "
                        "doesn't exists: %s" % extra_nrnmech_path)

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

bluepy_path = pathsconfig["BLUEPYPATH"]
if bluepy_path == '':
    try:
        import bluepy  # pylint: disable=C0413,C0411
    except ImportError:
        print 'Unable to import bluepy from default python path, \
                and BLUEPYPATH was not set during build either'
        raise
else:
    if not os.path.exists(bluepy_path):
        raise Exception('Inexistent BLUEPYPATH=%s' % bluepy_path)
    sys.path.insert(0, bluepy_path)
    try:
        import bluepy  # pylint: disable=C0413,C0411
    except ImportError:
        raise Exception('Unable to import bluepy from BLUEPYPATH=%s' %
                        bluepy_path)

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
