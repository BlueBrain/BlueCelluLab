"""Unit tests for the circuit_access module."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bglibpy.circuit import BluepyCircuitAccess, CellId, SonataCircuitAccess
from bglibpy.circuit.circuit_access import EmodelProperties, get_synapse_connection_parameters
from bglibpy.circuit import SynapseProperty
from bglibpy.exceptions import BGLibPyError
from tests.helpers.circuit import blueconfig_append_path


parent_dir = Path(__file__).resolve().parent.parent
proj55_path = "/gpfs/bbp.cscs.ch/project/proj55/"

test_thalamus_path = (
    Path(proj55_path) / "tuncel/simulations/release" / "2020-08-06-v2"
    / "bglibpy-thal-test-with-projections" / "BlueConfig")

hipp_circuit_with_projections = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


def test_non_existing_circuit_config():
    with pytest.raises(FileNotFoundError):
        BluepyCircuitAccess(str(parent_dir / "examples" / "non_existing_circuit_config"))


def test_get_synapse_connection_parameters():
    """Test that the synapse connection parameters are correctly created."""
    circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)
    # both cells are Excitatory
    pre_cell_id = CellId("hippocampus_neurons", 1)
    post_cell_id = CellId("hippocampus_neurons", 2)
    connection_params = get_synapse_connection_parameters(
        circuit_access, pre_cell_id, post_cell_id)
    assert connection_params["add_synapse"] is True
    assert connection_params["Weight"] == 1.0
    assert connection_params["SpontMinis"] == 0.01
    syn_configure = connection_params["SynapseConfigure"]
    assert syn_configure[0] == "%s.NMDA_ratio = 1.22 %s.tau_r_NMDA = 3.9 %s.tau_d_NMDA = 148.5"
    assert syn_configure[1] == "%s.mg = 1.0"


class TestCircuitAccess:

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        self.circuit_access = BluepyCircuitAccess(modified_conf)

    def test_target_contains_cell(self):
        target = "Mosaic"
        cell_id = CellId("", 1)
        assert self.circuit_access.target_contains_cell(target=target, cell_id=cell_id)

    def test_is_valid_group(self):
        group = "Mosaic"
        assert self.circuit_access.is_valid_group(group)

    def test_get_target_cell_ids(self):
        target = "Mosaic"
        res = self.circuit_access.get_target_cell_ids(target)
        assert res == {CellId("", 1), CellId("", 2)}

    def test_fetch_cell_info(self):
        cell_id = CellId("", 1)
        res = self.circuit_access.fetch_cell_info(cell_id)
        assert res.etype == "cADpyr"

        with pytest.raises(BGLibPyError):
            cell_id2 = CellId("", 99999999)
            self.circuit_access.fetch_cell_info(cell_id2)

    def test_fetch_mecombo_name(self):
        cell_id = CellId("", 1)
        res = self.circuit_access._fetch_mecombo_name(cell_id)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_emodel_name(self):
        cell_id = CellId("", 1)
        res = self.circuit_access._fetch_emodel_name(cell_id)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_mini_frequencies(self):
        cell_id = CellId("", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (None, None)

    @patch("bglibpy.circuit.BluepyCircuitAccess.fetch_cell_info")
    def test_fetch_mini_frequencies_with_mock_mvd(self, mock_fetch_cell_info):
        """Test fetching frequencies using MVD mini frequency keys."""
        mock_fetch_cell_info.return_value = pd.Series(
            {
                "exc_mini_frequency": 0.03,
                "inh_mini_frequency": 0.05,
            }
        )
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (0.03, 0.05)

    @patch("bglibpy.circuit.BluepyCircuitAccess.fetch_cell_info")
    def test_fetch_mini_frequencies_with_mock_sonata(self, mock_fetch_cell_info):
        """Test fetching frequencies using SONATA mini frequency keys."""
        mock_fetch_cell_info.return_value = pd.Series(
            {
                "exc-mini_frequency": 0.03,
                "inh-mini_frequency": 0.05,
            }
        )
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (0.03, 0.05)

    def test_node_properties_available(self):
        assert not self.circuit_access.node_properties_available

    def test_condition_parameters(self):
        res = self.circuit_access.config.condition_parameters()
        assert res.extracellular_calcium == 1.25
        assert res.randomize_gaba_rise_time is False
        assert res.mech_conditions.ampanmda.minis_single_vesicle == 1
        assert res.mech_conditions.gabaab.minis_single_vesicle == 1
        assert res.mech_conditions.glusynapse.minis_single_vesicle == 1
        assert res.mech_conditions.ampanmda.init_depleted == 1
        assert res.mech_conditions.gabaab.init_depleted == 1
        assert res.mech_conditions.glusynapse.init_depleted == 1

    def test_connection_entries(self):
        assert self.circuit_access.config.connection_entries() == []

    def test_is_glusynapse_used(self):
        assert not self.circuit_access.config.is_glusynapse_used

    def test_get_gids_of_mtypes(self):
        res = self.circuit_access.get_gids_of_mtypes(mtypes=["L5_TTPC1", "L6_TPC_L1"])
        assert res == {CellId("", 1), CellId("", 2)}

    def test_get_cell_ids_of_targets(self):
        target1 = "Mosaic"
        target2 = "Excitatory"
        res = self.circuit_access.get_cell_ids_of_targets(targets=[target1, target2])
        assert res == {CellId("", 1), CellId("", 2)}

    def test_morph_filepath(self):
        cell_id = CellId("", 1)
        res = self.circuit_access.morph_filepath(cell_id).rsplit("/", 1)[-1]
        assert res == "dend-C220197A-P2_axon-C060110A3_-_Clone_2.asc"

    def test_emodels_dir(self):
        res = self.circuit_access._emodels_dir
        assert Path(res).stem == "ccells"

    def test_emodel_path(self):
        cell_id = CellId("", 1)
        res = self.circuit_access.emodel_path(cell_id)
        fname = Path(res).name
        assert fname == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2.hoc"

    def test_extracellular_calcium(self):
        assert self.circuit_access.config.extracellular_calcium == 1.25

    def test_base_seed(self):
        assert self.circuit_access.config.base_seed == 12345

    def test_synapse_seed(self):
        assert self.circuit_access.config.synapse_seed == 0

    def test_ionchannel_seed(self):
        assert self.circuit_access.config.ionchannel_seed == 0

    def test_stimulus_seed(self):
        assert self.circuit_access.config.stimulus_seed == 0

    def test_minis_seed(self):
        assert self.circuit_access.config.minis_seed == 0

    def test_rng_mode(self):
        assert self.circuit_access.config.rng_mode == "Compatibility"

    def test_get_available_properties(self):
        assert self.circuit_access.available_cell_properties == {
            "mtype",
            "morph_class",
            "x",
            "y",
            "minicolumn",
            "etype",
            "morphology",
            "z",
            "orientation",
            "layer",
            "me_combo",
            "hypercolumn",
            "synapse_class",
        }

    def test_get_cell_properties(self):
        cell_id = CellId("", 1)
        coords = self.circuit_access.get_cell_properties(cell_id, properties=["x", "y", "z"])
        assert coords.x == 281.698701
        assert coords.y == 1035.578939
        assert coords.z == 671.566721

        etype = self.circuit_access.get_cell_properties(cell_id, properties="etype")
        assert etype.values[0] == "cADpyr"

    def test_get_emodel_properties(self):
        cell_id = CellId("", 1)
        res = self.circuit_access.get_emodel_properties(cell_id)
        assert res is None

    def test_get_template_format(self):
        res = self.circuit_access.get_template_format()
        assert res is None

    def test_spike_threshold(self):
        assert self.circuit_access.config.spike_threshold == -30.0

    def test_spike_location(self):
        assert self.circuit_access.config.spike_location == "soma"

    def test_duration(self):
        assert self.circuit_access.config.duration == 100.0

    def test_output_root_path(self):
        assert self.circuit_access.config.output_root_path == str(
            parent_dir / "examples" / "sim_twocell_all" / "output"
        )

    def test_dt(self):
        assert self.circuit_access.config.dt == 0.025

    def test_forward_skip(self):
        assert self.circuit_access.config.forward_skip is None

    def test_celsius(self):
        assert self.circuit_access.config.celsius == 34.0

    def test_v_init(self):
        assert self.circuit_access.config.v_init == -65.0

    def test_add_section(self):
        self.circuit_access.config.add_section(
            'Connection', 'SpontMinis_Exc',
            {
                'MorphologyPath': 'morph_path',
                'nrnPath': 'nrn_path',
                'METypePath': 'me_types',
                'TargetFile': 'target_file',
                'MorphologyType': 'swc',
            }
        )
        expected = (
            "Connection SpontMinis_Exc\n"
            "{\n"
            "  MorphologyPath morph_path\n"
            "  nrnPath nrn_path\n"
            "  METypePath me_types\n"
            "  TargetFile target_file\n"
            "  MorphologyType swc\n"
            "}\n"
        )
        assert str(self.circuit_access.config.impl).endswith(expected)


def test_get_connectomes_dict():
    """Test creation of connectome dict."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(conf_pre_path / "BlueConfig", conf_pre_path)
    circuit_access = BluepyCircuitAccess(modified_conf)

    connectomes_dict = circuit_access._get_connectomes_dict(None)
    assert connectomes_dict.keys() == {""}
    gid = 1
    assert connectomes_dict[""].afferent_synapses(gid) == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]


def test_extract_synapses():
    """Test extraction of synapses."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(conf_pre_path / "BlueConfig", conf_pre_path)
    circuit_access = BluepyCircuitAccess(modified_conf)
    projections = None
    all_properties = [
        SynapseProperty.PRE_GID,
        SynapseProperty.AXONAL_DELAY,
        SynapseProperty.POST_SECTION_ID,
        SynapseProperty.POST_SEGMENT_ID,
        SynapseProperty.POST_SEGMENT_OFFSET,
        SynapseProperty.G_SYNX,
        SynapseProperty.U_SYN,
        SynapseProperty.D_SYN,
        SynapseProperty.F_SYN,
        SynapseProperty.DTC,
        SynapseProperty.TYPE,
        SynapseProperty.NRRP,
        SynapseProperty.U_HILL_COEFFICIENT,
        SynapseProperty.CONDUCTANCE_RATIO]

    cell_id = CellId("", 1)
    synapses = circuit_access.extract_synapses(cell_id, all_properties, projections)

    proj_id, syn_idx = '', 0
    assert synapses.index[0] == (proj_id, syn_idx)
    assert synapses.iloc[0]["source_popid"] == 0.0
    assert synapses.iloc[0]["target_popid"] == 0.0
    assert synapses.shape == (5, 13)


def test_get_all_stimuli_entries():
    """Test creation of stimuli dict."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )
    circuit_access = BluepyCircuitAccess(modified_conf)

    all_stimuli = circuit_access.config.get_all_stimuli_entries()
    assert len(all_stimuli) == 2  # 3 -1 for spikereplay
    assert all_stimuli[0].target == "Excitatory"
    assert all_stimuli[1].duration == 20000


@pytest.mark.thal
def test_get_connectomes_dict_with_projections():
    """Test the retrieval of projection and the local connectomes."""
    circuit_access = BluepyCircuitAccess(str(test_thalamus_path))

    # empty
    assert circuit_access._get_connectomes_dict(None).keys() == {""}

    # single projection
    connectomes_dict = circuit_access._get_connectomes_dict("ML_afferents")
    assert connectomes_dict.keys() == {"", "ML_afferents"}

    # multiple projections
    all_connectomes = circuit_access._get_connectomes_dict(["ML_afferents", "CT_afferents"])
    assert all_connectomes.keys() == {"", "ML_afferents", "CT_afferents"}


@pytest.mark.thal
def test_get_all_projection_names():
    """Test the retrieval of projection and the local connectomes."""
    circuit_access = BluepyCircuitAccess(str(test_thalamus_path))

    assert set(circuit_access.config.get_all_projection_names()) == {
        "CT_afferents",
        "ML_afferents",
    }


@pytest.mark.thal
def test_get_population_ids():
    """Test the retrieval of population ids."""
    circuit_access = BluepyCircuitAccess(str(test_thalamus_path))
    assert circuit_access.get_population_ids("ML_afferents") == (1, 0)
    assert circuit_access.get_population_ids("CT_afferents") == (2, 0)


def test_sonata_circuit_access_file_not_found():
    with pytest.raises(FileNotFoundError):
        SonataCircuitAccess("non_existing_file")


class TestSonataCircuitAccess:
    def setup(self):
        self.circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)

    def test_available_cell_properties(self):
        assert self.circuit_access.available_cell_properties == {
            "x",
            "region",
            "rotation_angle_yaxis",
            "etype",
            "model_template",
            "synapse_class",
            "@dynamics:holding_current",
            "morph_class",
            "population",
            "morphology",
            "y",
            "layer",
            "mtype",
            "rotation_angle_xaxis",
            "z",
            "@dynamics:threshold_current",
            "model_type",
            "rotation_angle_zaxis",
        }

    def test_get_emodel_properties(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.get_emodel_properties(cell_id)
        assert res == EmodelProperties(
            threshold_current=0.33203125,
            holding_current=-0.116351104071555,
            ais_scaler=None,
        )

    def test_get_template_format(self):
        res = self.circuit_access.get_template_format()
        assert res == "v6"
        # if there was @dynamics:AIS_scaler, it would be v6_ais_scaler
        self.circuit_access.available_cell_properties.add("@dynamics:AIS_scaler")
        res = self.circuit_access.get_template_format()
        assert res == "v6_ais_scaler"

    def test_get_cell_properties(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.get_cell_properties(cell_id, "layer")
        assert res.values == ["SP"]

    def test_extract_synapses(self):
        cell_id = CellId("hippocampus_neurons", 1)
        projections = None
        all_properties = [
            SynapseProperty.PRE_GID,
            SynapseProperty.AXONAL_DELAY,
            SynapseProperty.POST_SECTION_ID,
            SynapseProperty.POST_SEGMENT_ID,
            SynapseProperty.POST_SEGMENT_OFFSET,
            SynapseProperty.G_SYNX,
            SynapseProperty.U_SYN,
            SynapseProperty.D_SYN,
            SynapseProperty.F_SYN,
            SynapseProperty.DTC,
            SynapseProperty.TYPE,
            SynapseProperty.NRRP,
            SynapseProperty.U_HILL_COEFFICIENT,
            SynapseProperty.CONDUCTANCE_RATIO]
        res = self.circuit_access.extract_synapses(cell_id, all_properties, projections)
        assert res.shape == (1742, 15)
        assert all(res["source_popid"] == 2126)

        assert all(res["source_population_name"] == "hippocampus_projections")
        assert all(res["target_popid"] == 378)
        assert all(res[SynapseProperty.POST_SEGMENT_ID] != -1)
        assert SynapseProperty.U_HILL_COEFFICIENT not in res.columns
        assert SynapseProperty.CONDUCTANCE_RATIO not in res.columns

        # projection parameter
        projection = "hippocampus_projections"
        res = self.circuit_access.extract_synapses(cell_id, all_properties, projection)
        assert res.shape == (1742, 15)
        list_of_single_projection = [projection]
        res = self.circuit_access.extract_synapses(cell_id, all_properties, list_of_single_projection)
        assert res.shape == (1742, 15)
        empty_projection = []
        res = self.circuit_access.extract_synapses(cell_id, all_properties, empty_projection)
        assert res.shape == (1742, 15)

    def test_target_contains_cell(self):
        target = "most_central_10_SP_PC"
        cell = CellId("hippocampus_neurons", 1)
        assert self.circuit_access.target_contains_cell(target, cell)
        cell2 = CellId("hippocampus_neurons", 100)
        assert not self.circuit_access.target_contains_cell(target, cell2)

    def test_get_target_cell_ids(self):
        target = "most_central_10_SP_PC"
        res = self.circuit_access.get_target_cell_ids(target)
        res_populations = {x.population_name for x in res}
        res_ids = {x.id for x in res}
        assert res_populations == {'hippocampus_neurons'}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_get_cell_ids_of_targets(self):
        targets = ["most_central_10_SP_PC", "Mosaic"]
        res = self.circuit_access.get_cell_ids_of_targets(targets)
        res_ids = {x.id for x in res}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_get_gids_of_mtypes(self):
        res = self.circuit_access.get_gids_of_mtypes(["SP_PC"])
        res_ids = {x.id for x in res}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_morph_filepath(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.morph_filepath(cell_id)
        fname = "dend-mpg141216_A_idA_axon-mpg141017_a1-2_idC"\
            "_-_Scale_x1.000_y0.900_z1.000_-_Clone_12.swc"
        assert res.rsplit("/", 1)[-1] == fname

    def test_emodel_path(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.emodel_path(cell_id)
        fname = "CA1_pyr_cACpyr_mpg141017_a1_2_idC_2019032814340.hoc"
        assert res.rsplit("/", 1)[-1] == fname

    def test_is_valid_group(self):
        group = "most_central_10_SP_PC"
        assert self.circuit_access.is_valid_group(group)
        group2 = "most_central_10_SP_PC2"
        assert not self.circuit_access.is_valid_group(group2)

    def test_fetch_cell_info(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_cell_info(cell_id)
        assert res.shape == (18,)
        assert res.etype == "cACpyr"
        assert res.mtype == "SP_PC"
        assert res.rotation_angle_zaxis == pytest.approx(-3.141593)

    def test_fetch_mini_frequencies(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (None, None)

    def test_node_properties_available(self):
        assert self.circuit_access.node_properties_available

    @patch("bglibpy.circuit.SonataCircuitAccess.fetch_cell_info")
    def test_fetch_mini_frequencies_with_mock(self, mock_fetch_cell_info):
        mock_fetch_cell_info.return_value = pd.Series(
            {
                "exc-mini_frequency": 0.03,
                "inh-mini_frequency": 0.05,
            }
        )
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (0.03, 0.05)

    def test_get_population_ids(self):
        edge_name = "hippocampus_projections__hippocampus_neurons__chemical"
        source_popid, target_popid = self.circuit_access.get_population_ids(edge_name)
        assert source_popid == 2126
        assert target_popid == 378


def test_morph_filepath_asc():
    """Unit test for SonataCircuitAccess::morph_filepath for asc retrieval."""
    circuit_sonata_quick_scx_config = (
        parent_dir
        / "examples"
        / "sonata_unit_test_sims"
        / "condition_parameters"
        / "simulation_config.json"
    )

    circuit_access = SonataCircuitAccess(circuit_sonata_quick_scx_config)
    asc_morph = circuit_access.morph_filepath(CellId("NodeA", 1))
    assert asc_morph.endswith(".asc")
