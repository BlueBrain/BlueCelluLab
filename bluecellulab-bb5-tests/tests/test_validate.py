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
import pathlib
from unittest.mock import patch

import pytest

from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit import BluepyCircuitAccess, SimulationValidator
from bluecellulab.exceptions import ConfigError, TargetDoesNotExist
from helpers.circuit import blueconfig_append_path

parent_dir = pathlib.Path(__file__).resolve().parent


class TestSimulationValidator:
    """Tests the parsing and evaluation of condition parameters."""

    def setup_method(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        circuit_access = BluepyCircuitAccess(modified_conf)
        self.sim_val = SimulationValidator(circuit_access)

    @patch("bluecellulab.circuit.config.BluepySimulationConfig.condition_parameters")
    def test_check_cao_cr_glusynapse_value(self, mock_cond_params):
        mock_cond_params.return_value = Conditions(
            mech_conditions=None, extracellular_calcium=99999999.9
        )
        assert self.sim_val.circuit_access.config.extracellular_calcium == 1.25
        with pytest.raises(ConfigError):
            self.sim_val.check_cao_cr_glusynapse_value()


def test_check_spike_location():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidSpikeLocation", conf_pre_path
    )
    circuit_access = BluepyCircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(ConfigError):
        sim_val.check_spike_location()


def test_check_connection_entries():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidConnectionEntries", conf_pre_path
    )
    circuit_access = BluepyCircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(TargetDoesNotExist):
        sim_val.check_connection_entries()
