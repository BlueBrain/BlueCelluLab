"""Unit tests for simulation/neuron_globals.py."""

import pathlib

import bglibpy
from bglibpy.circuit import CircuitAccess
from bglibpy.simulation import (
    set_minis_single_vesicle_values,
    set_global_condition_parameters,
    set_tstop_value
)
from tests.helpers.circuit import blueconfig_append_path


parent_dir = pathlib.Path(__file__).resolve().parent.parent


class TestConditionParameters:
    """Tests the parsing and evaluation of condition parameters."""

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        circuit_access = CircuitAccess(modified_conf)
        self.condition_parameters = circuit_access.config.condition_parameters_dict()
        self.h = bglibpy.neuron.h

    def test_set_global_condition_parameters(self):
        """Unit test for set_global_condition_parameters function."""
        set_global_condition_parameters(self.condition_parameters)
        init_depleted = self.condition_parameters["SYNAPSES__init_depleted"]
        assert self.h.init_depleted_GluSynapse == init_depleted
        assert self.h.init_depleted_ProbAMPANMDA_EMS == init_depleted
        assert self.h.init_depleted_ProbGABAAB_EMS == init_depleted

    def test_set_minis_single_vesicle_values(self):
        """Unit test for set_minis_single_vesicle_values."""
        minis_single_vesicle = int(self.condition_parameters["SYNAPSES__minis_single_vesicle"])
        set_minis_single_vesicle_values(minis_single_vesicle)
        assert self.h.minis_single_vesicle_ProbAMPANMDA_EMS == minis_single_vesicle
        assert self.h.minis_single_vesicle_ProbGABAAB_EMS == minis_single_vesicle
        assert self.h.minis_single_vesicle_GluSynapse == minis_single_vesicle

    def test_set_tstop_value(self):
        """Unit test for setting global tstop value."""
        tstop = 55
        set_tstop_value(tstop)
        assert self.h.tstop == tstop
