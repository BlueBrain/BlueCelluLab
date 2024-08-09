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
"""Unit tests for simulation/neuron_globals.py."""

import pathlib

import pytest

import bluecellulab
from bluecellulab.circuit import BluepyCircuitAccess
from bluecellulab.circuit.config.sections import ConditionEntry, MechanismConditions
from bluecellulab.importer import _load_mod_files
from bluecellulab.simulation import (
    set_minis_single_vesicle_values,
    set_global_condition_parameters,
)
from helpers.circuit import blueconfig_append_path


parent_dir = pathlib.Path(__file__).resolve().parent
_load_mod_files()


class TestConditionParameters:
    """Tests the parsing and evaluation of condition parameters."""

    def setup_method(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        circuit_access = BluepyCircuitAccess(modified_conf)
        self.condition_parameters = circuit_access.config.condition_parameters()
        self.h = bluecellulab.neuron.h

    @pytest.mark.v6
    def test_set_global_condition_parameters(self):
        """Unit test for set_global_condition_parameters function."""
        set_global_condition_parameters(self.condition_parameters)
        mech_conditions = self.condition_parameters.mech_conditions
        init_depleted_glusynapse = mech_conditions.glusynapse.init_depleted
        assert self.h.init_depleted_GluSynapse == init_depleted_glusynapse
        init_depleted_ampanmda = mech_conditions.ampanmda.init_depleted
        assert self.h.init_depleted_ProbAMPANMDA_EMS == init_depleted_ampanmda
        init_depleted_gabaab = mech_conditions.gabaab.init_depleted
        assert self.h.init_depleted_ProbGABAAB_EMS == init_depleted_gabaab

    @pytest.mark.v6
    def test_set_minis_single_vesicle_values(self):
        """Unit test for set_minis_single_vesicle_values."""
        mech_conditions = MechanismConditions(
            ampanmda=ConditionEntry(minis_single_vesicle=1)
        )
        set_minis_single_vesicle_values(mech_conditions)
        assert self.h.minis_single_vesicle_ProbAMPANMDA_EMS == 1
        assert self.h.minis_single_vesicle_ProbGABAAB_EMS == 0
        assert self.h.minis_single_vesicle_GluSynapse == 0
