#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulation class of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import sys
from bglibpy.importer import neuron
from bglibpy import printv
from bglibpy import printv_err
from bglibpy import tools

class Simulation(object):

    """Class that represents a neuron simulation"""
    def __init__(self, show_progress=False):
        self.cells = []
        self.fih = None
        if show_progress:
            self.progress = neuron.h.ShowProgress( neuron.h.cvode, 0 )
        else:
            self.progress = None

    def add_cell(self, new_cell):
        """Add a cell to a simulation"""
        self.cells.append(new_cell)



    def init_callbacks(self):
        """Initialize the callback of all the registered simulation objects (e.g. for window plotting)"""
        for cell in self.cells:
            cell.init_callbacks()
        #if not self.progress and bglibpy.VERBOSE_LEVEL > 0:
        #    self.progress = neuron.h.ShowProgress( neuron.h.cvode, 0 )

    def run(self, maxtime, cvode=True, celsius=34, v_init=-65, dt=0.025):
        """Run the simulation"""
        neuron.h.celsius = celsius
        neuron.h.tstop = maxtime

        cvode_old_status = neuron.h.cvode_active()
        if cvode:
            neuron.h.cvode_active(1)
        else:
            if cvode_old_status:
                printv("WARNING: cvode was activated outside of Simulation, temporarily disabling it in run() because cvode=False was set", 2)
            neuron.h.cvode_active(0)

        neuron.h.v_init = v_init

        for cell in self.cells:
            try:
                cell.re_init_rng()
            except AttributeError:
                sys.exc_clear()

        neuron.h.dt = dt
        neuron.h.steps_per_ms = 1.0/dt

        """
        WARNING: Be 'very' careful when you change something below.

        This can easily break the BGLib replay, since the way things are initialized heavily influence the random number generator
        e.g. finitialize() + step() != run()
        """

        printv('Running a simulation until %f ms ...' % maxtime, 1)

        self.init_callbacks()

        #pylint: disable=W0703
        try:
            neuron.h.run()
        except Exception, e:
            printv_err('The neuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e), 1)
        finally:
            if cvode_old_status:
                printv("WARNING: cvode was activated outside of Simulation, this might make it impossible to load templates with stochastic channels", 2)
            neuron.h.cvode_active(cvode_old_status)

        printv('Finished simulation', 1)

    def __del__(self):
        pass

    @tools.deprecated
    def addCell(self, new_cell):
        """Add a cell to a simulation"""
        self.add_cell(new_cell)

