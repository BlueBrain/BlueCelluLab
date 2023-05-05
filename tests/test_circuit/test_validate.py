"""Unit tests for circuit/validate.py."""

import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from bglibpy.circuit import BluepyCircuitAccess, SimulationValidator
from bglibpy.circuit.config.sections import Conditions
from bglibpy.circuit.synapse_properties import SynapseProperty
from bglibpy.circuit.validate import check_nrrp_value
from bglibpy.exceptions import ConfigError, TargetDoesNotExist
from tests.helpers.circuit import blueconfig_append_path


parent_dir = pathlib.Path(__file__).resolve().parent.parent


class TestSimulationValidator:
    """Tests the parsing and evaluation of condition parameters."""

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        circuit_access = BluepyCircuitAccess(modified_conf)
        self.sim_val = SimulationValidator(circuit_access)

    @patch("bglibpy.circuit.config.BluepySimulationConfig.condition_parameters")
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


def test_check_nrrp_value():
    """Unit test for check nrrp value."""
    synapses = pd.DataFrame(data={SynapseProperty.NRRP: [15.0, 16.0]})

    check_nrrp_value(synapses)

    synapses[SynapseProperty.NRRP].loc[0] = 15.1
    with pytest.raises(ValueError):
        check_nrrp_value(synapses)

    synapses[SynapseProperty.NRRP].loc[0] = -1

    with pytest.raises(ValueError):
        check_nrrp_value(synapses)
