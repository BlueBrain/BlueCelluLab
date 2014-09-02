# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for Cell.py"""

import nose.tools as nt
import math
import bglibpy
import os
from nose.plugins.attrib import attr

class TestCellBaseClass1(object):

    """First Cell test class"""

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "examples/cell_example1/test_cell.hoc",
            "examples/cell_example1")
        nt.assert_true(isinstance(self.cell, bglibpy.Cell))

    def teardown(self):
        """Teardown"""
        del self.cell

    def test_fields(self):
        """Cell: Test the fields of a Cell object"""
        nt.assert_true(isinstance(self.cell.soma, bglibpy.neuron.nrn.Section))
        nt.assert_true(
            isinstance(
                self.cell.axonal[0],
                bglibpy.neuron.nrn.Section))
        nt.assert_true(math.fabs(self.cell.threshold - 0.184062) < 0.00001)
        nt.assert_true(math.fabs(self.cell.hypamp - -0.070557) < 0.00001)
        nt.assert_true(math.fabs(self.cell.soma.diam - 13.78082) < 0.00001)
        nt.assert_true(math.fabs(self.cell.soma.L - 19.21902) < 0.00001)
        nt.assert_true(math.fabs(self.cell.basal[2].diam - 0.595686) < 0.00001)
        nt.assert_true(math.fabs(self.cell.basal[2].L - 178.96164) < 0.00001)
        nt.assert_true(
            math.fabs(
                self.cell.apical[10].diam -
                0.95999) < 0.00001)
        nt.assert_true(math.fabs(self.cell.apical[10].L - 23.73195) < 0.00001)

    def test_addRecording(self):
        """Cell: Test if addRecording gives deprecation warning"""
        import warnings
        warnings.simplefilter('default')
        varname = 'self.apical[1](0.5)._ref_v'
        with warnings.catch_warnings(record=True) as w:
            self.cell.addRecording(varname)
            nt.assert_true(
                len([warning for warning in w
                     if issubclass(warning.category, DeprecationWarning)]) > 0)

    def test_get_hsection(self):
        """Cell: Test cell.get_hsection"""
        nt.assert_true(
            isinstance(
                self.cell.get_hsection(0),
                bglibpy.neuron.nrn.Section))

    def test_add_recording(self):
        """Cell: Test cell.add_recording"""
        varname = 'self.apical[1](0.5)._ref_v'
        self.cell.add_recording(varname)
        nt.assert_true(varname in self.cell.recordings)

    def test_add_recordings(self):
        """Cell: Test cell.add_recordings"""
        varnames = [
            'self.axonal[0](0.25)._ref_v',
            'self.soma(0.5)._ref_v',
            'self.apical[1](0.5)._ref_v']
        self.cell.add_recordings(varnames)
        for varname in varnames:
            nt.assert_true(varname in self.cell.recordings)

    def test_add_allsections_voltagerecordings(self):
        """Cell: Test cell.add_allsections_voltagerecordings"""
        self.cell.add_allsections_voltagerecordings()

        all_sections = self.cell.cell.getCell().all
        for section in all_sections:
            varname = 'neuron.h.%s(0.5)._ref_v' % section.name()
            nt.assert_true(varname in self.cell.recordings)


@attr('bgscratch')
class TestCellBaseClassSynapses(object):

    """TestCellBaseClassSynapses"""

    def setup(self):
        """Setup TestCellBaseClassSynapses"""
        self.prev_cwd = os.getcwd()
        os.chdir("examples/sim_twocell_synapseid")
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2)

    def test_info_dict(self):
        """Cell: Test if info_dict is working as expected"""

        import pickle

        with open('cell1_info_dict.pickle', 'r') as cell_info_dict_file:
            cell1_info_dict_expected = pickle.load(cell_info_dict_file)

        cell1_info_dict = self.ssim_bglibpy.cells[1].info_dict
        nt.assert_equal(cell1_info_dict, cell1_info_dict_expected)

    def teardown(self):
        """Teardown TestCellBaseClassSynapses"""
        os.chdir(self.prev_cwd)
        del self.ssim_bglibpy
