#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose.tools as nt
import math
import bglibpy

class TestCellBaseClass(object):
    def setup(self):
        self.cell = bglibpy.Cell("test_cell/test_cell.hoc", "test_cell")
        self.sim = bglibpy.Simulation()
        nt.assert_true(isinstance(self.sim, bglibpy.Simulation))

    def teardown(self):
        del self.cell
        del self.sim


    def test_run(self):
        self.sim.run(100)
