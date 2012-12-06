"""
Python library for running single cell bglib templates
"""

import sys
import os
import numpy
import matplotlib as plt
import pylab

installdir = os.path.dirname(__file__)
pathsconfig_filename = installdir + "/paths.config"

if os.path.exists(pathsconfig_filename):
    pathsconfig = dict((line.strip().split("=")[0], line.strip().split("=")[1]) for line in open(pathsconfig_filename, "r"))
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
neuron.h.load_file("SerializedCell.hoc")
neuron.h.load_file("TStim.hoc")
neuron.h('obfunc new_IClamp() { return new IClamp($1) }')

from cell import Cell
'''
from plotwindow import PlotWindow
from dendrogram import Dendrogram
from psection import PSection
from psegment import PSegment
from simulation import Simulation
'''
