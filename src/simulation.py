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

class Simulation:

    """Class that represents a neuron simulation"""
    def __init__(self, verbose_level=0):
        self.verbose_level = verbose_level
        self.cells = []

    def addCell(self, new_cell):
        """Add a cell to a simulation"""
        self.cells.append(new_cell)

    def run(self, maxtime, cvode=True, celsius=34, v_init=-65, dt=0.025):
        """Run the simulation"""
        neuron.h.celsius = celsius
        neuron.h.tstop = maxtime

        cvode_old_status = neuron.h.cvode_active()
        if cvode:
            neuron.h.cvode_active(1)
        else:
            if cvode_old_status:
                printv("WARNING: cvode was activated outside of Simulation, temporarily disabling it in run() because cvode=True was set", 2)
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

        #neuron.h.finitialize()
        printv('Running a simulation until %f ms ...' % maxtime, 1)

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

        #self.continuerun(maxtime)

    #def continuerun(self, maxtime):
    #    """Continue a running simulation"""
    #    while neuron.h.t < maxtime:
    #        for cell in self.cells:
    #            cell.update()
    #        if self.verbose_level >= 1:
    #            print str(neuron.h.t) + " ms"
            #try:
    #        neuron.h.step()
            #except Exception, e:
            #    print 'The neuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e)
            #    break

    def __del__(self):
        pass
