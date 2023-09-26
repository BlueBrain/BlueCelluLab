from unittest import mock
import pytest
import bluecellulab
from bluecellulab.circuit.config.sections import ConditionEntry, Conditions, MechanismConditions
from bluecellulab.simulation.neuron_globals import set_global_condition_parameters, set_init_depleted_values, set_minis_single_vesicle_values, set_tstop_value


def test_set_tstop_value():
    set_tstop_value(100.0)
    assert bluecellulab.neuron.h.tstop == 100.0


@mock.patch("bluecellulab.neuron.h")
def test_set_global_condition_parameters_no_mechs(mocked_h):
    del mocked_h.cao_CR_GluSynapse  # delete the attribute for the mock
    calcium_condition = Conditions(extracellular_calcium=2.0)
    set_global_condition_parameters(calcium_condition)
    assert not hasattr(bluecellulab.neuron.h, "cao_CR_GluSynapse")

    mech_conditions = Conditions(mech_conditions=MechanismConditions(None, None, None))
    set_global_condition_parameters(mech_conditions)


@pytest.mark.v6
def test_set_global_condition_parameters():
    calcium_condition = Conditions(extracellular_calcium=2.015)
    set_global_condition_parameters(calcium_condition)
    assert bluecellulab.neuron.h.cao_CR_GluSynapse == 2.015


@pytest.mark.v6
def test_set_init_depleted_values():
    mech_conditions = MechanismConditions(
        ampanmda=ConditionEntry(minis_single_vesicle=None, init_depleted=True),
        gabaab=ConditionEntry(minis_single_vesicle=None, init_depleted=False),
        glusynapse=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
    )
    set_init_depleted_values(mech_conditions)
    assert bluecellulab.neuron.h.init_depleted_ProbAMPANMDA_EMS == 1.0
    assert bluecellulab.neuron.h.init_depleted_ProbGABAAB_EMS == 0.0
    assert bluecellulab.neuron.h.init_depleted_GluSynapse == 0.0


@pytest.mark.v6
def test_set_minis_single_vesicle_values():
    mech_conditions = MechanismConditions(
        ampanmda=ConditionEntry(minis_single_vesicle=True, init_depleted=None),
        gabaab=ConditionEntry(minis_single_vesicle=False, init_depleted=None),
        glusynapse=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
    )
    set_minis_single_vesicle_values(mech_conditions)
    assert bluecellulab.neuron.h.minis_single_vesicle_ProbAMPANMDA_EMS == 1.0
    assert bluecellulab.neuron.h.minis_single_vesicle_ProbGABAAB_EMS == 0.0
    assert bluecellulab.neuron.h.minis_single_vesicle_GluSynapse == 0.0
