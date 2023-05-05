"""Unit tests for simulation/neuron_globals.py."""

import pathlib

import pytest

import bglibpy
from bglibpy.circuit import BluepyCircuitAccess
from bglibpy.circuit.config.sections import ConditionEntry, MechanismConditions
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
        circuit_access = BluepyCircuitAccess(modified_conf)
        self.condition_parameters = circuit_access.config.condition_parameters()
        self.h = bglibpy.neuron.h

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
        mech_conditions = MechanismConditions(ampanmda=ConditionEntry(minis_single_vesicle=1))
        set_minis_single_vesicle_values(mech_conditions)
        assert self.h.minis_single_vesicle_ProbAMPANMDA_EMS == 1
        assert self.h.minis_single_vesicle_ProbGABAAB_EMS == 0
        assert self.h.minis_single_vesicle_GluSynapse == 0

    @pytest.mark.v6
    def test_set_tstop_value(self):
        """Unit test for setting global tstop value."""
        tstop = 55
        set_tstop_value(tstop)
        assert self.h.tstop == tstop
