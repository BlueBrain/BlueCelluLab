# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for Cell.py"""

import math
import os
import random
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import bglibpy
from bglibpy.cell.template import NeuronTemplate, shorten_and_hash_string
from bglibpy.exceptions import BGLibPyError
from tests.helpers.circuit import blueconfig_append_path

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

script_dir = os.path.dirname(__file__)


def test_longname():
    """Cell: Test loading cell with long name"""

    cell = bglibpy.Cell(
        "%s/examples/cell_example1/test_cell_longname1.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir)
    assert isinstance(cell, bglibpy.Cell)

    del cell


def test_load_template():
    """Test the neuron template loading."""
    fpath = Path(script_dir) / "examples/cell_example1/test_cell.hoc"
    template_name = NeuronTemplate.load(fpath)
    assert template_name == "test_cell_bglibpy"


@patch("bglibpy.neuron")
def test_load_template_with_old_neuron(mock_bglibpy_neuron):
    """Test the template loading with an old neuron version."""
    mock_bglibpy_neuron.h.nrnversion.return_value = '2014-03-19'
    fpath = Path(script_dir) / "examples/cell_example1/test_cell.hoc"
    template_name = NeuronTemplate.load(fpath)
    assert template_name == "test_cell"
    mock_bglibpy_neuron.h.nrnversion.assert_called_once_with(4)
    mock_bglibpy_neuron.h.load_file.assert_called_once()
    mock_bglibpy_neuron.h.assert_not_called()


def test_shorten_and_hash_string():
    """Unit test for the shorten_and_hash_string function."""
    with pytest.raises(ValueError):
        shorten_and_hash_string(label="1", hash_length=21)

    short_label = "short-label"
    assert shorten_and_hash_string(short_label) == short_label

    long_label = "test-cell" * 10
    assert len(shorten_and_hash_string(long_label)) < len(long_label)


class TestCellBaseClass1:
    """First Cell test class"""

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % script_dir,
            "%s/examples/cell_example1" % script_dir)
        assert isinstance(self.cell, bglibpy.Cell)

    def teardown(self):
        """Teardown"""
        del self.cell

    def test_fields(self):
        """Cell: Test the fields of a Cell object"""
        assert isinstance(self.cell.soma, bglibpy.neuron.nrn.Section)
        assert isinstance(self.cell.axonal[0], bglibpy.neuron.nrn.Section)
        assert math.fabs(self.cell.threshold - 0.184062) < 0.00001
        assert math.fabs(self.cell.hypamp - -0.070557) < 0.00001
        # Lowered precision because of commit
        # 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn simulator repo
        assert math.fabs(self.cell.soma.diam - 13.78082) < 0.0001
        assert math.fabs(self.cell.soma.L - 19.21902) < 0.00001
        assert math.fabs(self.cell.basal[2].diam - 0.595686) < 0.00001
        assert math.fabs(self.cell.basal[2].L - 178.96164) < 0.00001
        assert math.fabs(self.cell.apical[10].diam - 0.95999) < 0.00001
        assert math.fabs(self.cell.apical[10].L - 23.73195) < 0.00001

    def test_get_hsection(self):
        """Cell: Test cell.get_hsection"""
        assert isinstance(
            self.cell.get_hsection(0), bglibpy.neuron.nrn.Section)

    def test_add_recording(self):
        """Cell: Test cell.add_recording"""
        varname = 'self.apical[1](0.5)._ref_v'
        self.cell.add_recording(varname)
        assert varname in self.cell.recordings

        second_varname = 'self.apical[1](0.6)._ref_v'
        self.cell.add_recording(second_varname, dt=0.025)
        assert second_varname in self.cell.recordings

    def test_add_recordings(self):
        """Cell: Test cell.add_recordings"""
        varnames = [
            'self.axonal[0](0.25)._ref_v',
            'self.soma(0.5)._ref_v',
            'self.apical[1](0.5)._ref_v']
        self.cell.add_recordings(varnames)
        for varname in varnames:
            assert varname in self.cell.recordings

    def test_add_allsections_voltagerecordings(self):
        """Cell: Test cell.add_allsections_voltagerecordings"""
        self.cell.add_allsections_voltagerecordings()

        all_sections = self.cell.cell.getCell().all
        for section in all_sections:
            varname = 'neuron.h.%s(0.5)._ref_v' % section.name()
            assert varname in self.cell.recordings

    def test_manual_add_allsection_voltage_recordings(self):
        """Cell: Test cell.add_voltage_recording."""
        all_sections = self.cell.cell.getCell().all
        last_section = None
        for section in all_sections:
            self.cell.add_voltage_recording(section, 0.5)
            recording = self.cell.get_voltage_recording(section, 0.5)
            assert len(recording) == 0
            last_section = section
        with pytest.raises(BGLibPyError):
            self.cell.get_voltage_recording(last_section, 0.7)

    def test_euclid_section_distance(self):
        """Cell: Test cell.euclid_section_distance"""

        random.seed(1)

        location1 = 0.0
        location2 = 1.0
        for _ in range(1000):
            hsection1 = random.choice(random.choice(
                [self.cell.apical, self.cell.somatic, self.cell.basal]))
            hsection2 = random.choice(random.choice(
                [self.cell.apical, self.cell.somatic, self.cell.basal]))
            distance_euclid = \
                self.cell.euclid_section_distance(hsection1=hsection1,
                                                  hsection2=hsection2,
                                                  location1=location1,
                                                  location2=location2,
                                                  projection='xyz')

            x1 = bglibpy.neuron.h.x3d(0,
                                      sec=hsection1)
            y1 = bglibpy.neuron.h.y3d(0,
                                      sec=hsection1)
            z1 = bglibpy.neuron.h.z3d(0,
                                      sec=hsection1)
            x2 = bglibpy.neuron.h.x3d(
                bglibpy.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            y2 = bglibpy.neuron.h.y3d(
                bglibpy.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            z2 = bglibpy.neuron.h.z3d(
                bglibpy.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            import numpy as np
            distance_hand = np.sqrt((x1 - x2) ** 2
                                    + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            assert distance_euclid == distance_hand


class TestCellBaseClassSynapses:

    """TestCellBaseClassSynapses"""

    def setup(self):
        """Setup TestCellBaseClassSynapses"""
        self.gid = 1

        self.conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_synapseid")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(self.conf_pre_path, "BlueConfig"), self.conf_pre_path)

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            synapse_detail=2)

    def test_info_dict(self):
        """Cell: Test if info_dict is working as expected"""

        cell1_info_dict = self.ssim_bglibpy.cells[self.gid].info_dict

        import pprint
        cell1_info_dict_str = pprint.pformat(cell1_info_dict)

        # with open('cell1_info_dict.txt', 'w') as cell_info_dict_file:
        #    cell_info_dict_file.write(cell1_info_dict_str)

        expected_dict = os.path.join(self.conf_pre_path, "cell1_info_dict.txt")

        with open(expected_dict, 'r') as cell_info_dict_file:
            cell1_info_dict_expected_str = cell_info_dict_file.read()

        # print(cell1_info_dict_str)

        # print(cell1_info_dict_expected_str)
        assert cell1_info_dict_str == cell1_info_dict_expected_str

    def teardown(self):
        """Teardown TestCellBaseClassSynapses"""
        self.ssim_bglibpy.delete()
        assert (bglibpy.tools.check_empty_topology())


@pytest.mark.debugtest
class TestCellBaseClassVClamp:

    """First Cell test class"""

    def setup(self):
        """Setup"""
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % script_dir,
            "%s/examples/cell_example1" % script_dir)
        assert (isinstance(self.cell, bglibpy.Cell))

    def teardown(self):
        """Teardown"""
        del self.cell

    def test_add_voltage_clamp(self):
        """Cell: Test add_voltage_clamp"""

        level = -90
        stop_time = 50
        total_time = 200
        rs = .1
        vclamp = self.cell.add_voltage_clamp(
            stop_time=stop_time,
            level=level,
            current_record_name='test_vclamp',
            rs=rs)

        assert vclamp.amp1 == level
        assert vclamp.dur1 == stop_time
        assert vclamp.dur2 == 0
        assert vclamp.dur3 == 0
        assert vclamp.rs == rs

        sim = bglibpy.Simulation()
        sim.add_cell(self.cell)
        sim.run(total_time, dt=.1, cvode=False)

        time = self.cell.get_time()
        current = self.cell.get_recording('test_vclamp')
        import numpy as np

        voltage = self.cell.get_soma_voltage()

        voltage_vc_end = np.mean(
            voltage[np.where((time < stop_time) & (time > .9 * stop_time))])

        assert (abs(voltage_vc_end - level) < .1)

        voltage_end = np.mean(
            voltage
            [np.where((time < total_time) & (time > .9 * total_time))])

        assert (abs(voltage_end - (-73)) < 1)

        current_vc_end = np.mean(
            current[np.where((time < stop_time) & (time > .9 * stop_time))])

        assert (abs(current_vc_end - (-.1)) < .01)

        current_after_vc_end = np.mean(
            current[
                np.where((time > stop_time) & (time < 1.1 * stop_time))])

        assert current_after_vc_end == 0.0


def test_get_recorded_spikes():
    """Cell: Test get_recorded_spikes."""
    cell = bglibpy.Cell(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir)
    sim = bglibpy.Simulation()
    sim.add_cell(cell)
    cell.start_recording_spikes(None, "soma", -30)
    cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
    sim.run(24, cvode=False)
    spikes = cell.get_recorded_spikes("soma")
    ground_truth = [3.350000000100014, 11.52500000009988, 19.9750000000994]
    assert np.allclose(spikes, ground_truth)
