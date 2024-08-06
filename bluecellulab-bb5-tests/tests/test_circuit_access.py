# Copyright 2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the circuit_access module."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bluecellulab.circuit import BluepyCircuitAccess, CellId
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.circuit.config.sections import ConnectionOverrides

from helpers.circuit import blueconfig_append_path

parent_dir = Path(__file__).resolve().parent
proj55_path = "/gpfs/bbp.cscs.ch/project/proj55/"

test_thalamus_path = (
    Path(proj55_path)
    / "tuncel/simulations/release"
    / "2020-08-06-v2"
    / "bglibpy-thal-test-with-projections"
    / "BlueConfig"
)


def test_non_existing_circuit_config():
    with pytest.raises(FileNotFoundError):
        BluepyCircuitAccess(
            str(parent_dir / "examples" / "non_existing_circuit_config")
        )


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
    all_connectomes = circuit_access._get_connectomes_dict(
        ["ML_afferents", "CT_afferents"]
    )
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


class TestCircuitAccess:
    def setup_method(self):
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

        with pytest.raises(BluecellulabError):
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

    @patch("bluecellulab.circuit.BluepyCircuitAccess.fetch_cell_info")
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

    @patch("bluecellulab.circuit.BluepyCircuitAccess.fetch_cell_info")
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

    def test_connection_override(self):
        entries = self.circuit_access.config.connection_entries()
        assert len(entries) == 0

        connection_override = ConnectionOverrides(
            source="Excitatory",
            target="Mosaic",
            delay=2,
            weight=2.0,
            spont_minis=0.1,
            synapse_configure="%s.mg = 1.4",
            mod_override=None,
        )
        self.circuit_access.config.add_connection_override(connection_override)

        entries = self.circuit_access.config.connection_entries()
        assert len(entries) == 1
        assert entries[-1] == connection_override

        # overrides are not added multiple times
        entries = self.circuit_access.config.connection_entries()
        entries = self.circuit_access.config.connection_entries()
        assert len(entries) == 1

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
        assert (
            fname
            == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2.hoc"
        )

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
        coords = self.circuit_access.get_cell_properties(
            cell_id, properties=["x", "y", "z"]
        )
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

    cell_id = CellId("", 1)
    synapses = circuit_access.extract_synapses(cell_id, projections)

    proj_id, syn_idx = "", 0
    assert synapses.index[0] == (proj_id, syn_idx)
    assert synapses.iloc[0]["source_popid"] == 0.0
    assert synapses.iloc[0]["target_popid"] == 0.0
    assert synapses.shape[0] == 5


def test_get_all_stimuli_entries():
    """Test creation of stimuli dict."""
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(conf_pre_path / "BlueConfig", conf_pre_path)
    circuit_access = BluepyCircuitAccess(modified_conf)

    all_stimuli = circuit_access.config.get_all_stimuli_entries()
    assert len(all_stimuli) == 2  # 3 -1 for spikereplay
    assert all_stimuli[0].target == "Excitatory"
    assert all_stimuli[1].duration == 20000
