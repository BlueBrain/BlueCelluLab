#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Nosetest for the Simulation class"""

import numpy

import nose.tools as nt
import bglibpy


class TestCellBaseClass(object):

    """Base test class"""

    def __init__(self):
        """Initializer"""
        self.cell = None
        self.sim = None

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "examples/cell_example1/test_cell.hoc",
            "examples/cell_example1")
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        nt.assert_true(isinstance(self.sim, bglibpy.Simulation))

    def teardown(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run the simulation for 100 ms"""
        self.sim.run(100)


class TestCellcSTUTRandom123BaseClass(object):

    """Base test class"""

    def __init__(self):
        """Initializer"""
        self.cell = None
        self.sim = None

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "examples/cell_example_cstut/cSTUT_7.hoc",
            "examples/cell_example_cstut")
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        nt.assert_true(isinstance(self.sim, bglibpy.Simulation))

    def teardown(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run a simulation with cSTUT and Random123 for 300 ms"""
        self.sim.run(300, cvode=False, use_random123_stochkv=True)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[numpy.where(time > 150)]

        nt.assert_almost_equal(numpy.mean(voltage_ss), -75.5400762008)
        nt.assert_almost_equal(numpy.std(voltage_ss), 0.142647101877)


class TestCellcSTUTBaseClass(object):

    """Base test class"""

    def __init__(self):
        """Initializer"""
        self.cell = None
        self.sim = None

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "examples/cell_example_cstut/cSTUT_7.hoc",
            "examples/cell_example_cstut")
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)
        nt.assert_true(isinstance(self.sim, bglibpy.Simulation))

    def teardown(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run a simulation with cSTUT and MCellRan4 for 300 ms"""
        self.sim.run(300, cvode=False, use_random123_stochkv=False)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[numpy.where(time > 150)]

        print numpy.mean(voltage_ss)
        print numpy.std(voltage_ss)

        nt.assert_almost_equal(numpy.mean(voltage_ss), -75.61918061202924)
        nt.assert_almost_equal(numpy.std(voltage_ss), 0.19192736450671288)
