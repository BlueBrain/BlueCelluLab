#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Nosetest for the Cell class"""

import nose.tools as nt
#import math
import bglibpy

class TestCellBaseClass(object):
    """Base test class"""

    def __init__(self):
        """Initializer"""
        self.cell = None
        self.sim = None

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell("examples/cell_example1/test_cell.hoc", "examples/cell_example1")
        self.sim = bglibpy.Simulation()
        nt.assert_true(isinstance(self.sim, bglibpy.Simulation))

    def teardown(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Run the simulation for 100 ms"""
        self.sim.run(100)
