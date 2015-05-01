#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulation class of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import sys
import bglibpy
from bglibpy.importer import neuron
from bglibpy import printv
from bglibpy import printv_err
from bglibpy import tools


class Simulation(object):

    """Class that represents a neuron simulation"""

    def __init__(self):
        self.cells = []
        self.fih_progress = None
        self.progress = None
        self.progress_closed = None
        self.progress_dt = None

    def add_cell(self, new_cell):
        """Add a cell to a simulation"""
        self.cells.append(new_cell)

    def init_progress_callback(self):
        """Initiziale the progress bar callback."""
        self.progress = 0
        if not self.fih_progress:
            self.fih_progress = neuron.h.FInitializeHandler(
                1, self.progress_callback)
        self.progress_closed = False

    def progress_callback(self):
        """Callback function for the progress bar"""
        if self.progress > 0:
            sys.stdout.write("\x1b[3F")

        self.progress += 1
        self.progress_closed = False if self.progress_closed else True
        if self.progress_closed:
            sys.stdout.write(" %s%s%s \n" % (" " * (
                self.progress - 1), " ", " " * (100 - self.progress)))
            sys.stdout.write("[%s%s%s]\n" % ("#" * (
                self.progress - 1), "-", "." * (100 - self.progress)))
            sys.stdout.write(" %s%s%s \n" % (" " * (
                self.progress - 1), " ", " " * (100 - self.progress)))
        else:
            sys.stdout.write(" %s%s%s \n" % (" " * (
                self.progress - 1), "/", " " * (100 - self.progress)))
            sys.stdout.write("[%s%s%s]\n" % ("#" * (
                self.progress - 1), ">", "." * (100 - self.progress)))
            sys.stdout.write(" %s%s%s \n" % (" " * (
                self.progress - 1), "\\", " " * (100 - self.progress)))
        sys.stdout.flush()

        neuron.h.cvode.event(
            neuron.h.t + self.progress_dt, self.progress_callback)

    def init_callbacks(self):
        """Initialize the callback of all the registered simulation objects.

           (e.g. for window plotting)
        """
        for cell in self.cells:
            cell.init_callbacks()

    # pylint: disable=C0103,R0912
    def run(self, maxtime, cvode=True, celsius=34, v_init=-65, dt=0.025,
            forward_skip=None, forward_skip_value=False, show_progress=None):
        """Run the simulation."""
        # if maxtime <= neuron.h.t:
        #     raise Exception("Simulation: need to provide a maxtime (=%f) "
        #                    "that is bigger than the current time (=%f)" % (
        #                        maxtime, neuron.h.t))

        if show_progress is None:
            if bglibpy.VERBOSE_LEVEL > 1:
                show_progress = True
            else:
                show_progress = False

        if show_progress:
            self.progress_dt = maxtime / 100
            self.init_progress_callback()

        neuron.h.celsius = celsius
        neuron.h.tstop = maxtime

        cvode_old_status = neuron.h.cvode_active()
        if cvode:
            neuron.h.cvode_active(1)
        else:
            if cvode_old_status:
                printv(
                    "WARNING: cvode was activated outside of Simulation, "
                    "temporarily disabling it in run() because cvode=False "
                    "was set", 2)
            neuron.h.cvode_active(0)

        neuron.h.v_init = v_init

        for cell in self.cells:
            try:
                cell.re_init_rng()
            except AttributeError:
                sys.exc_clear()

        neuron.h.dt = dt
        neuron.h.steps_per_ms = 1.0 / dt

        # WARNING: Be 'very' careful when you change something below.

        # This can easily break the BGLib replay, since the way things are
        # initialized heavily influence the random number generator
        # e.g. finitialize() + step() != run()

        printv('Running a simulation until %f ms ...' % maxtime, 1)

        self.init_callbacks()

        neuron.h.stdinit()

        if forward_skip_value is not None and forward_skip:
            neuron.h.t = -1e9
            save_dt = neuron.h.dt
            neuron.h.dt = forward_skip_value * 0.1
            for _ in range(0, 10):
                neuron.h.fadvance()
            neuron.h.dt = save_dt
            neuron.h.t = 0

        # pylint: disable=W0703
        try:
            neuron.h.continuerun(neuron.h.tstop)
        except Exception as exception:
            printv_err("The neuron was eaten by the Python !\n"
                       "Reason: % s: % s" % (
                           exception.__class__.__name__, exception), 1)
        finally:
            if cvode_old_status:
                printv(
                    "WARNING: cvode was activated outside of Simulation, "
                    "this might make it impossible to load templates with "
                    "stochastic channels", 2)
            neuron.h.cvode_active(cvode_old_status)

        printv('Finished simulation', 1)

    # pylint: enable=C0103,R0912

    def __del__(self):
        pass

    @tools.deprecated
    # pylint: disable=C0103
    def addCell(self, new_cell):
        """Add a cell to a simulation"""
        self.add_cell(new_cell)
    # pylint: enable=C0103
