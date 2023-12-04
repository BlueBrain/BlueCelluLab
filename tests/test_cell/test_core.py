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
from bluecellulab.ssim import SSim

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


@pytest.mark.v5
def test_load_template():
    """Test the neuron template loading."""
    hoc_path = parent_dir / "examples/cell_example1/test_cell.hoc"
    morph_path = parent_dir / "examples/cell_example1/test_cell.asc"
    template = NeuronTemplate(hoc_path, morph_filepath=morph_path)
    template_name = template.template_name
    assert template_name == f"test_cell_bluecellulab_{hex(id(template))}"


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

    def test_add_ais_recording(self):
        """Cell Test add_ais_recording."""
        self.cell.add_ais_recording()
        ais_key = "self.axonal[1](0.5)._ref_v"
        assert ais_key in self.cell.recordings

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


class TestCellSpikes:

    def setup(self):
        self.cell = bluecellulab.Cell(
            f"{parent_dir}/examples/cell_example1/test_cell.hoc",
            f"{parent_dir}/examples/cell_example1")
        self.sim = bluecellulab.Simulation()
        self.sim.add_cell(self.cell)

    @pytest.mark.v5
    def test_get_recorded_spikes(self):
        """Cell: Test get_recorded_spikes."""
        self.cell.start_recording_spikes(None, "soma", -30)
        self.cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
        self.sim.run(24, cvode=False)
        spikes = self.cell.get_recorded_spikes("soma")
        ground_truth = [3.350000000100014, 11.52500000009988, 19.9750000000994]
        assert np.allclose(spikes, ground_truth)

    @pytest.mark.v5
    def test_create_netcon_spikedetector(self):
        """Cell: create_netcon_spikedetector."""
        threshold = -29.0
        netcon = self.cell.create_netcon_spikedetector(None, "AIS", -29.0)
        assert netcon.threshold == threshold
        netcon = self.cell.create_netcon_spikedetector(None, "soma", -29.0)
        with pytest.raises(ValueError):
            self.cell.create_netcon_spikedetector(None, "Dendrite", -29.0)


@pytest.mark.v6
def test_add_dendrogram():
    """Cell: Test get_recorded_spikes."""
    emodel_properties = EmodelProperties(threshold_current=1.1433533430099487,
                                         holding_current=1.4146618843078613,
                                         AIS_scaler=1.4561502933502197,
                                         soma_scaler=1.0)
    cell = bluecellulab.Cell(
        "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
        "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
        template_format="v6",
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
class TestCellV6:
    """Test class for testing Cell object functionalities with v6 template."""

    def setup(self):
        """Setup."""
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0
        )
        self.cell = bluecellulab.Cell(
            "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
            "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
            template_format="v6",
            emodel_properties=emodel_properties
        )

    def test_repr_and_str(self):
        """Test the repr and str representations of a cell object."""
        # >>> print(cell)
        # Cell Object: <bluecellulab.cell.core.Cell object at 0x7f73b3fb2550>.
        # NEURON ID: cADpyr_L2TPC_bluecellulab_0x7f73b48e2510.
        # make sure NEURON template name is in the string representation
        assert self.cell.cell.hname().split('[')[0] in str(self.cell)

    def test_get_section_id(self):
        """Test the get_section_id method."""
        self.cell.init_psections()
        assert self.cell.get_section_id(str(self.cell.soma)) == 0
        assert self.cell.get_section_id(str(self.cell.axonal[0])) == 1
        assert self.cell.get_section_id(str(self.cell.basal[0])) == 145
        assert self.cell.get_section_id(str(self.cell.apical[0])) == 169

    def test_area(self):
        """Test the cell's area computation."""
        assert self.cell.area() == 5812.493415302344

    def test_cell_id(self):
        """Test for checking if cell_id is different btw. 2 cells when unspecified."""
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0
        )
        cell2 = bluecellulab.Cell(
            "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
            "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
            template_format="v6",
            emodel_properties=emodel_properties
        )
        assert self.cell.cell_id != cell2.cell_id

    def test_get_childrensections(self):
        """Test the get_childrensections method."""
        res = self.cell.get_childrensections(self.cell.soma)
        assert len(res) == 3

    def test_get_parentsection(self):
        """Test the get_parentsection method."""
        section = self.cell.get_childrensections(self.cell.soma)[0]
        res = self.cell.get_parentsection(section)
        assert res == self.cell.soma


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


@pytest.mark.v6
class TestWithinCircuit:

    def setup(self):
        """Setup method called before each test method."""
        sonata_sim_path = (
            parent_dir
            / "examples"
            / "sim_quick_scx_sonata_multicircuit"
            / "simulation_config_noinput.json"
        )
        cell_id = ("NodeA", 0)
        ssim = SSim(sonata_sim_path)
        ssim.instantiate_gids(cell_id, add_synapses=True, add_stimuli=False)
        self.cell = ssim.cells[cell_id]
        self.ssim = ssim  # for persistance

    def test_pre_gids(self):
        """Test get_pre_gids within a circuit."""
        pre_gids = self.cell.pre_gids()
        assert pre_gids == [0, 1]

    def test_pre_gid_synapse_ids(self):
        """Test pre_gid_synapse_ids within a circuit."""
        pre_gids = self.cell.pre_gids()

        first_gid_synapse_ids = self.cell.pre_gid_synapse_ids(pre_gids[0])
        assert first_gid_synapse_ids == [('NodeB__NodeA__chemical', 0), ('NodeB__NodeA__chemical', 2)]

        second_gid_synapse_ids = self.cell.pre_gid_synapse_ids(pre_gids[1])
        assert len(second_gid_synapse_ids) == 4
        assert second_gid_synapse_ids[0] == ('NodeA__NodeA__chemical', 0)
        assert second_gid_synapse_ids[1] == ('NodeA__NodeA__chemical', 1)
        assert second_gid_synapse_ids[2] == ('NodeB__NodeA__chemical', 1)
        assert second_gid_synapse_ids[3] == ('NodeB__NodeA__chemical', 3)
