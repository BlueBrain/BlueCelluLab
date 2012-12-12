#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose.tools as nt
import math
import bglibpy

class TestCellBaseClass(object):
    def setup(self):
        self.cell = bglibpy.Cell("test_cell/test_cell.hoc", "test_cell")
        nt.assert_true(isinstance(self.cell, bglibpy.Cell))

    def teardown(self):
        del self.cell

    def test_soma(self):
        nt.assert_true(isinstance(self.cell.soma, bglibpy.neuron.nrn.Section))

    def test_get_section(self):
        nt.assert_true(isinstance(self.cell.get_section(0), bglibpy.neuron.nrn.Section))

    def test_get_threshold(self):
        nt.assert_true(math.fabs(self.cell.get_threshold() - 0.184062) < 0.00001)

    def test_get_hypamp(self):
        nt.assert_true(math.fabs(self.cell.get_hypamp() - -0.070557) < 0.00001)
