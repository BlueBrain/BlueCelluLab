#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Tests for the Simulation class"""

import os
import numpy
import bglibpy

from pytest import approx

script_dir = os.path.dirname(__file__)


class TestCellBaseClass:

    """Base test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % script_dir,
            "%s/examples/cell_example1" % script_dir)
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        assert isinstance(self.sim, bglibpy.Simulation)

    def teardown_method(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run the simulation for 100 ms"""
        self.sim.run(100)


class TestCellcSTUTRandom123BaseClass:

    """Base test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example_cstut/cSTUT_7.hoc" % script_dir,
            "%s/examples/cell_example_cstut" % script_dir)
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        assert isinstance(self.sim, bglibpy.Simulation)

    def teardown_method(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run a simulation with cSTUT and Random123 for 300 ms"""
        self.sim.run(300, cvode=False, use_random123_stochkv=True)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[numpy.where(time > 150)]

        # Lowered precision because of
        # commit 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn repo
        # self.cell = bglibpy.Cell("%s/examples/cell_example_cstut/cSTUT_7.hoc" % script_dir,"%s/examples/cell_example_cstut" % script_dir)
        assert numpy.mean(voltage_ss) == approx(-75.5400762008, abs=1e-6)
        assert numpy.std(voltage_ss) == approx(0.142647101877, abs=1e-6)


class TestCellcSTUTBaseClass:

    """Base test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example_cstut/cSTUT_7.hoc" % script_dir,
            "%s/examples/cell_example_cstut" % script_dir)
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        assert isinstance(self.sim, bglibpy.Simulation)

    def teardown_method(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run a simulation with cSTUT and MCellRan4 for 300 ms"""
        self.sim.run(300, cvode=False, use_random123_stochkv=False)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[numpy.where(time > 150)]

        # Lowered precision because of
        # commit 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn repo
        assert numpy.mean(voltage_ss) == approx(-75.61918061202924, abs=1e-6)
        assert numpy.std(voltage_ss) == approx(0.19192736450671288, abs=1e-6)
