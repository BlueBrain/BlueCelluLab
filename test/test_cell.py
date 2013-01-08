#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose.tools as nt
import math
import bglibpy

class TestCellBaseClass(object):
    def setup(self):
        self.cell = bglibpy.Cell("cell_example1/test_cell.hoc", "cell_example1")
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

    def test_add_recording(self):
        varname = 'self.apical[1](0.5)._ref_v'
        self.cell.add_recording(varname)
        nt.assert_true(varname in self.cell.recordings)

    def test_add_recordings(self):
        varnames = ['self.axonal[0](0.25)._ref_v', 'self.soma(0.5)._ref_v', 'self.apical[1](0.5)._ref_v']
        self.cell.add_recordings(varnames)
        for varname in varnames:
                nt.assert_true(varname in self.cell.recordings)

    def test_add_allsections_voltagerecordings(self):
        varname = 'neuron.h.Cell[0].apic[10](0.5)._ref_v'
        self.cell.add_allsections_voltagerecordings()
        nt.assert_true(varname in self.cell.recordings)
