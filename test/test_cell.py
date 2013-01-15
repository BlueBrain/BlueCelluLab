#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for Cell.py"""

import nose.tools as nt
import math
import bglibpy

class TestCellBaseClass1(object):
    """First Cell test class"""
    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell("examples/cell_example1/test_cell.hoc", "examples/cell_example1")
        nt.assert_true(isinstance(self.cell, bglibpy.Cell))

    def teardown(self):
        """Teardown"""
        del self.cell

    def test_fields(self):
        """Test the fields of a Cell object"""
        nt.assert_true(isinstance(self.cell.soma, bglibpy.neuron.nrn.Section))
        nt.assert_true(isinstance(self.cell.axonal[0], bglibpy.neuron.nrn.Section))
        nt.assert_true(math.fabs(self.cell.threshold - 0.184062) < 0.00001)
        nt.assert_true(math.fabs(self.cell.hypamp - -0.070557) < 0.00001)

    def test_addRecording(self):
        """Test if addRecording gives deprecation warning"""
        import warnings
        warnings.simplefilter('default')
        varname = 'self.apical[1](0.5)._ref_v'
        with warnings.catch_warnings(record=True) as w:
            self.cell.addRecording(varname)
            nt.assert_true(len(filter(lambda i: issubclass(i.category, DeprecationWarning), w)) > 0)

    def test_get_section(self):
        """Test cell.get_section"""
        nt.assert_true(isinstance(self.cell.get_section(0), bglibpy.neuron.nrn.Section))

    def test_add_recording(self):
        """Test cell.add_recording"""
        varname = 'self.apical[1](0.5)._ref_v'
        self.cell.add_recording(varname)
        nt.assert_true(varname in self.cell.recordings)

    def test_add_recordings(self):
        """Test cell.add_recordings"""
        varnames = ['self.axonal[0](0.25)._ref_v', 'self.soma(0.5)._ref_v', 'self.apical[1](0.5)._ref_v']
        self.cell.add_recordings(varnames)
        for varname in varnames:
            nt.assert_true(varname in self.cell.recordings)

    def test_add_allsections_voltagerecordings(self):
        """Test cell.add_allsections_voltagerecordings"""
        varname = 'neuron.h.Cell[0].apic[10](0.5)._ref_v'
        self.cell.add_allsections_voltagerecordings()
        nt.assert_true(varname in self.cell.recordings)
