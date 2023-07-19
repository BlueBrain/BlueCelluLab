# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for Cell.py"""

import math
import random
import warnings
from pathlib import Path
from bluecellulab.circuit.circuit_access import EmodelProperties

import numpy as np
import pytest

import bluecellulab
from bluecellulab.cell.template import NeuronTemplate, shorten_and_hash_string
from bluecellulab.exceptions import BluecellulabError

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v5
def test_longname():
    """Cell: Test loading cell with long name"""

    cell = bluecellulab.Cell(
        "%s/examples/cell_example1/test_cell_longname1.hoc" % str(parent_dir),
        "%s/examples/cell_example1" % str(parent_dir))
    assert isinstance(cell, bluecellulab.Cell)

    del cell


def test_load_template():
    """Test the neuron template loading."""
    fpath = parent_dir / "examples/cell_example1/test_cell.hoc"
    template_name = NeuronTemplate.load(fpath)
    assert template_name == "test_cell_bluecellulab"


def test_shorten_and_hash_string():
    """Unit test for the shorten_and_hash_string function."""
    with pytest.raises(ValueError):
        shorten_and_hash_string(label="1", hash_length=21)

    short_label = "short-label"
    assert shorten_and_hash_string(short_label) == short_label

    long_label = "test-cell" * 10
    assert len(shorten_and_hash_string(long_label)) < len(long_label)


@pytest.mark.v5
class TestCellBaseClass1:
    """First Cell test class"""

    def setup(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
            "%s/examples/cell_example1" % str(parent_dir))
        assert isinstance(self.cell, bluecellulab.Cell)

    def teardown(self):
        """Teardown"""
        del self.cell

    def test_fields(self):
        """Cell: Test the fields of a Cell object"""
        assert isinstance(self.cell.soma, bluecellulab.neuron.nrn.Section)
        assert isinstance(self.cell.axonal[0], bluecellulab.neuron.nrn.Section)
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
            self.cell.get_hsection(0), bluecellulab.neuron.nrn.Section)

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
        with pytest.raises(BluecellulabError):
            self.cell.get_voltage_recording(last_section, 0.7)

    def test_get_allsections_voltagerecordings(self):
        """Cell: Test cell.get_allsections_voltagerecordings."""
        self.cell.recordings.clear()

        with pytest.raises(BluecellulabError):
            recordings = self.cell.get_allsections_voltagerecordings()

        self.cell.add_allsections_voltagerecordings()
        recordings = self.cell.get_allsections_voltagerecordings()
        assert len(recordings) == len(self.cell.recordings)
        for recording in recordings:
            assert any(recording in s for s in self.cell.recordings)

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

            x1 = bluecellulab.neuron.h.x3d(0,
                                           sec=hsection1)
            y1 = bluecellulab.neuron.h.y3d(0,
                                           sec=hsection1)
            z1 = bluecellulab.neuron.h.z3d(0,
                                           sec=hsection1)
            x2 = bluecellulab.neuron.h.x3d(
                bluecellulab.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            y2 = bluecellulab.neuron.h.y3d(
                bluecellulab.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            z2 = bluecellulab.neuron.h.z3d(
                bluecellulab.neuron.h.n3d(
                    sec=hsection2) - 1,
                sec=hsection2)
            import numpy as np
            distance_hand = np.sqrt((x1 - x2) ** 2
                                    + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            assert distance_euclid == distance_hand


@pytest.mark.debugtest
class TestCellBaseClassVClamp:

    """First Cell test class"""

    def setup(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
            "%s/examples/cell_example1" % str(parent_dir))
        assert (isinstance(self.cell, bluecellulab.Cell))

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

        sim = bluecellulab.Simulation()
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


@pytest.mark.v5
def test_get_recorded_spikes():
    """Cell: Test get_recorded_spikes."""
    cell = bluecellulab.Cell(
        "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
        "%s/examples/cell_example1" % str(parent_dir))
    sim = bluecellulab.Simulation()
    sim.add_cell(cell)
    cell.start_recording_spikes(None, "soma", -30)
    cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
    sim.run(24, cvode=False)
    spikes = cell.get_recorded_spikes("soma")
    ground_truth = [3.350000000100014, 11.52500000009988, 19.9750000000994]
    assert np.allclose(spikes, ground_truth)


@pytest.mark.v6
def test_add_dendrogram():
    """Cell: Test get_recorded_spikes."""
    emodel_properties = EmodelProperties(threshold_current=1.1433533430099487,
                                         holding_current=1.4146618843078613,
                                         ais_scaler=1.4561502933502197)
    cell = bluecellulab.Cell(
        "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
        "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
        template_format="v6_ais_scaler",
        emodel_properties=emodel_properties)
    cell.add_plot_window(['self.soma(0.5)._ref_v'])
    output_path = "cADpyr_L2TPC_dendrogram.png"
    cell.add_dendrogram(save_fig_path=output_path)
    sim = bluecellulab.Simulation()
    sim.add_cell(cell)
    cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
    sim.run(24, cvode=False)
    assert Path(output_path).is_file()


@pytest.mark.v6
def test_add_synapse_replay():
    """Cell: test add_synapse_replay."""
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sonata_unit_test_sims"
        / "synapse_replay"
        / "simulation_config.json"
    )
    ssim = bluecellulab.SSim(sonata_sim_path)
    ssim.spike_threshold = -900.0
    cell_id = ("hippocampus_neurons", 0)
    ssim.instantiate_gids(cell_id,
                          add_stimuli=True, add_synapses=True,
                          interconnect_cells=False)
    cell = ssim.cells[cell_id]
    assert len(cell.connections) == 3
    assert cell.connections[
        ("hippocampus_projections__hippocampus_neurons__chemical", 0)
    ].pre_spiketrain.tolist() == [16.0, 22.0, 48.0]
