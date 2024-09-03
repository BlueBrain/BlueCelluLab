# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulation class of bluecellulab."""

import contextlib
import sys
from typing import Optional
import logging

import neuron

import bluecellulab

logger = logging.getLogger(__name__)


class Simulation:
    """Class that represents a neuron simulation."""

    def __init__(self, parallel_context=None, custom_progress_function=None) -> None:
        self.cells: list[bluecellulab.Cell] = []
        self.fih_progress = None
        self.progress = None
        self.progress_closed = None
        self.progress_dt: Optional[float] = None
        self.pc = parallel_context
        self.custom_progress_function = custom_progress_function

    def add_cell(self, new_cell: bluecellulab.Cell) -> None:
        """Add a cell to a simulation."""
        self.cells.append(new_cell)

    def init_progress_callback(self):
        """Initiziale the progress bar callback."""
        self.progress = 0
        if not self.fih_progress:
            self.fih_progress = neuron.h.FInitializeHandler(
                1, self.progress_callback)
        self.progress_closed = False

    def progress_callback(self):
        """Callback function for the progress bar."""
        if self.custom_progress_function is None:
            if self.progress > 0:
                sys.stdout.write("\x1b[3F")

            self.progress += 1
            self.progress_closed = not self.progress_closed
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
        else:
            self.custom_progress_function()

        neuron.h.cvode.event(
            neuron.h.t + self.progress_dt, self.progress_callback)

    def init_callbacks(self):
        """Initialize the callback of all the registered simulation objects.

        (e.g. for window plotting)
        """
        for cell in self.cells:
            cell.init_callbacks()

    def run(
            self,
            maxtime: float,
            cvode=True,
            cvode_minstep=None,
            cvode_maxstep=None,
            dt=0.025,
            forward_skip=None,
            forward_skip_value=False,
            show_progress=None
    ):
        """Run the simulation."""
        # if maxtime <= neuron.h.t:
        #     raise Exception("Simulation: need to provide a maxtime (=%f) "
        #                    "that is bigger than the current time (=%f)" % (
        #                        maxtime, neuron.h.t))

        if show_progress is None:
            show_progress = bluecellulab.VERBOSE_LEVEL > 1

        if show_progress:
            self.progress_dt = maxtime / 100
            self.init_progress_callback()

        neuron.h.tstop = maxtime

        cvode_old_status = neuron.h.cvode_active()
        if cvode:
            neuron.h.cvode_active(1)
            if cvode_minstep:
                neuron.h.cvode.minstep(cvode_minstep)
            if cvode_maxstep:
                neuron.h.cvode.maxstep(cvode_maxstep)
        else:
            if cvode_old_status:
                logger.warning("cvode was activated outside of Simulation, "
                               "temporarily disabling it in run() because cvode=False "
                               "was set")
            neuron.h.cvode_active(0)

        for cell in self.cells:
            with contextlib.suppress(AttributeError):
                cell.re_init_rng()
        neuron.h.dt = dt
        neuron.h.steps_per_ms = 1.0 / dt

        # WARNING: Be 'very' careful when you change something below.

        # This can easily break the BGLib replay, since the way things are
        # initialized heavily influence the random number generator
        # e.g. finitialize() + step() != run()

        logger.debug(f'Running a simulation until {maxtime} ms ...')

        self.init_callbacks()

        neuron.h.stdinit()

        if forward_skip:
            if forward_skip_value is not None and forward_skip_value > 0:
                neuron.h.t = -1e9
                save_dt = neuron.h.dt
                neuron.h.dt = forward_skip_value * 0.1
                for _ in range(0, 10):
                    neuron.h.fadvance()
                neuron.h.dt = save_dt
                neuron.h.t = 0.0

        if self.pc is not None:
            for cell in self.cells:
                self.pc.prcellstate(cell.cell_id.id, f"bluecellulab_t={neuron.h.t}")

        try:
            neuron.h.continuerun(neuron.h.tstop)
        except Exception as exception:
            stream_handler = logging.StreamHandler(sys.stderr)
            logger.addHandler(stream_handler)
            logger.error("The neuron was eaten by the Python !\n"
                         "Reason: % s: % s" % (exception.__class__.__name__, exception))
            logger.removeHandler(stream_handler)
        finally:
            if cvode_old_status:
                logger.warning(
                    "cvode was activated outside of Simulation, "
                    "this might make it impossible to load templates with "
                    "stochastic channels")
            neuron.h.cvode_active(cvode_old_status)

        logger.debug("Finished simulation.")
