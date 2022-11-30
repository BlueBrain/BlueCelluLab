"""Unit tests for the circuit_access module."""

from pathlib import Path

import pytest

from bglibpy.circuit import CircuitAccess
from bglibpy.exceptions import BGLibPyError
from tests.helpers.circuit import blueconfig_append_path


parent_dir = Path(__file__).resolve().parent.parent
proj55_path = "/gpfs/bbp.cscs.ch/project/proj55/"

test_thalamus_path = (
    Path(proj55_path) / "tuncel/simulations/release" / "2020-08-06-v2"
    / "bglibpy-thal-test-with-projections" / "BlueConfig")


def test_non_existing_circuit_config():
    with pytest.raises(FileNotFoundError):
        CircuitAccess(str(parent_dir / "examples" / "non_existing_circuit_config"))


class TestCircuitAccess:

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        self.circuit_access = CircuitAccess(modified_conf)

    def test_use_mecombo_tsv(self):
        assert not self.circuit_access.use_mecombo_tsv

    def test_is_cell_target(self):
        assert self.circuit_access.is_cell_target(target="a1", gid=1)

    def test_is_group_target(self):
        assert self.circuit_access.is_group_target(target="Mosaic")

    def test_get_target_gids(self):
        assert self.circuit_access._get_target_gids(target="Mosaic") == {1, 2}

    def test_target_has_gid(self):
        assert self.circuit_access.target_has_gid(target="Mosaic", gid=1)

    def test_fetch_gid_cell_info(self):
        res = self.circuit_access.fetch_gid_cell_info(gid=1)
        assert res.etype == "cADpyr"

        with pytest.raises(BGLibPyError):
            self.circuit_access.fetch_gid_cell_info(gid=99999999)

    def test_fetch_mecombo_name(self):
        res = self.circuit_access.fetch_mecombo_name(gid=1)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_emodel_name(self):
        res = self.circuit_access.fetch_emodel_name(gid=1)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_morph_name(self):
        res = self.circuit_access.fetch_morph_name(gid=1)
        assert res == "dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_mini_frequencies(self):
        res = self.circuit_access.fetch_mini_frequencies(gid=1)
        assert res == (None, None)

    def test_node_properties_available(self):
        assert not self.circuit_access.node_properties_available

    def test_condition_parameters_dict(self):
        res = self.circuit_access.config.condition_parameters_dict()
        assert res["cao_CR_GluSynapse"] == 1.25

    def test_connection_entries(self):
        assert self.circuit_access.config.connection_entries == []

    def test_is_glusynapse_used(self):
        assert not self.circuit_access.config.is_glusynapse_used

    def test_get_gids_of_mtypes(self):
        res = self.circuit_access.get_gids_of_mtypes(mtypes=["L5_TTPC1", "L6_TPC_L1"])
        assert res == {1, 2}

    def test_get_gids_of_targets(self):
        res = self.circuit_access.get_gids_of_targets(targets=["Mosaic", "Excitatory"])
        assert res == {1, 2}

    def test_get_morph_dir_and_extension(self):
        f_dir, f_ext = self.circuit_access.config.get_morph_dir_and_extension()
        assert Path(f_dir).stem == "ascii"
        assert f_ext == "asc"

    def test_morph_dir(self):
        res = self.circuit_access.config.morph_dir
        assert Path(res).stem == "ascii"

    def test_morph_extension(self):
        res = self.circuit_access.config.morph_extension
        assert res == "asc"

    def test_morph_filename(self):
        res = self.circuit_access.morph_filename(gid=1)
        assert res == "dend-C220197A-P2_axon-C060110A3_-_Clone_2.asc"

    def test_emodels_dir(self):
        res = self.circuit_access.config.emodels_dir
        assert Path(res).stem == "ccells"

    def test_emodel_path(self):
        res = self.circuit_access.emodel_path(gid=1)
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
        coords = self.circuit_access.get_cell_properties(gid=1, properties=["x", "y", "z"])
        assert coords.x == 281.698701
        assert coords.y == 1035.578939
        assert coords.z == 671.566721

        etype = self.circuit_access.get_cell_properties(gid=1, properties="etype")
        assert etype.values[0] == "cADpyr"

    def test_get_emodel_info(self):
        res = self.circuit_access.get_emodel_info(gid=1)
        assert res == {
            "combo_name": "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"
        }

    def test_get_cell_ids(self):
        assert self.circuit_access.get_cell_ids("PreCell") == {2}

    def test_spike_threshold(self):
        assert self.circuit_access.config.spike_threshold == -30.0

    def test_spike_location(self):
        assert self.circuit_access.config.spike_location == "soma"

    def test_duration(self):
        assert self.circuit_access.config.duration == 100.0

    def test_deprecated_minis_single_vesicle(self):
        assert self.circuit_access.config.deprecated_minis_single_vesicle is None

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
        expected = '''\
Connection SpontMinis_Exc
{
  MorphologyPath morph_path
  nrnPath nrn_path
  METypePath me_types
  TargetFile target_file
  MorphologyType swc
}
'''
        assert str(self.circuit_access.config.bc).endswith(expected)


def test_get_connectomes_dict():
    """Test creation of connectome dict."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)

    connectomes_dict = circuit_access.get_connectomes_dict(None)
    assert connectomes_dict.keys() == {""}


def test_get_all_stimuli_entries():
    """Test creation of stimuli dict."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)

    all_stimuli = circuit_access.config.get_all_stimuli_entries()
    assert len(all_stimuli) == 3
    assert all_stimuli[0]["Mode"] == "Current"
    assert all_stimuli[0]["Target"] == "Mosaic"


@pytest.mark.thal
def test_get_connectomes_dict_with_projections():
    """Test the retrieval of projection and the local connectomes."""
    circuit_access = CircuitAccess(test_thalamus_path)

    # empty
    assert circuit_access.get_connectomes_dict(None).keys() == {""}

    # single projection
    connectomes_dict = circuit_access.get_connectomes_dict("ML_afferents")
    assert connectomes_dict.keys() == {"", "ML_afferents"}

    # multiple projections
    all_connectomes = circuit_access.get_connectomes_dict(["ML_afferents", "CT_afferents"])
    assert all_connectomes.keys() == {"", "ML_afferents", "CT_afferents"}


@pytest.mark.thal
def test_get_all_projection_names():
    """Test the retrieval of projection and the local connectomes."""
    circuit_access = CircuitAccess(test_thalamus_path)

    assert set(circuit_access.config.get_all_projection_names()) == {
        "CT_afferents",
        "ML_afferents",
    }


@pytest.mark.thal
def test_get_population_ids():
    """Test the retrieval of population ids."""
    circuit_access = CircuitAccess(test_thalamus_path)

    assert circuit_access.config.get_population_ids(
        ignore_populationid_error=True, projection="ML_afferents"
    ) == (1, 0)
    assert circuit_access.config.get_population_ids(
        ignore_populationid_error=True, projection="CT_afferents"
    ) == (2, 0)
