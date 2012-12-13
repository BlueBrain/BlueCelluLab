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

    def test_fields(self):
        nt.assert_true(isinstance(self.cell.soma, bglibpy.neuron.nrn.Section))
        nt.assert_true(isinstance(self.cell.axonal[0], bglibpy.neuron.nrn.Section))
        nt.assert_true(math.fabs(self.cell.threshold - 0.184062) < 0.00001)
        nt.assert_true(math.fabs(self.cell.hypamp - -0.070557) < 0.00001)

    def test_get_section(self):
        nt.assert_true(isinstance(self.cell.get_section(0), bglibpy.neuron.nrn.Section))
