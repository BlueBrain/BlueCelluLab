"""Unit tests for circuit/validate.py."""

import pathlib
from unittest import mock
from unittest.mock import patch

import pytest

import bglibpy
from bglibpy.circuit import CircuitAccess, SimulationValidator


parent_dir = pathlib.Path(__file__).resolve().parent.parent


class TestSimulationValidator:
    """Tests the parsing and evaluation of condition parameters."""

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = bglibpy.tools.blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        circuit_access = CircuitAccess(modified_conf)
        self.sim_val = SimulationValidator(circuit_access)

    @patch("bglibpy.neuron")
    def test_check_single_vesicle_minis_settings(self, mock_bglibpy_neuron):
        # make sure there are no other attributes than the_only_attribute
        mock_bglibpy_neuron.h = mock.Mock(["the_only_attribute"])
        with pytest.raises(bglibpy.OldNeurodamusVersionError):
            self.sim_val.check_single_vesicle_minis_settings()

    @patch("bglibpy.circuit.CircuitAccess.condition_parameters_dict")
    def test_check_randomize_gaba_risetime(self, mock_cond_params):
        mock_cond_params.return_value = {"randomize_Gaba_risetime": "InBetweenTrueAndFalse"}
        with pytest.raises(bglibpy.ConfigError):
            self.sim_val.check_randomize_gaba_risetime()

    @patch("bglibpy.circuit.CircuitAccess.condition_parameters_dict")
    def test_check_cao_cr_glusynapse_value(self, mock_cond_params):
        mock_cond_params.return_value = {"cao_CR_GluSynapse": "999999999"}
        assert self.sim_val.circuit_access.extracellular_calcium == 1.25
        with pytest.raises(bglibpy.ConfigError):
            self.sim_val.check_cao_cr_glusynapse_value()


def test_check_spike_location():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidSpikeLocation", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(bglibpy.ConfigError):
        sim_val.check_spike_location()


def test_check_mod_override_file():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidModOverride", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(bglibpy.ConfigError):
        sim_val.check_mod_override_file()


def test_check_connection_entries_with_invalid_connection_content():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidConnectionContents", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(bglibpy.ConfigError):
        sim_val.check_connection_entries()


def test_check_connection_entries():
    conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfigWithInvalidConnectionEntries", conf_pre_path
    )
    circuit_access = CircuitAccess(modified_conf)
    sim_val = SimulationValidator(circuit_access)
    with pytest.raises(bglibpy.TargetDoesNotExist):
        sim_val.check_connection_entries()
