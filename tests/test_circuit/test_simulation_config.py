"""Unit tests for circuit/config/simulation_config.py."""

from pathlib import Path

from bluepysnap import Simulation as SnapSimulation
import pytest

from bluecellulab.circuit.config import SonataSimulationConfig
from bluecellulab.circuit.config.sections import (
    ConditionEntry,
    Conditions,
    ConnectionOverrides,
    MechanismConditions,
)
from bluecellulab.stimuli import Noise, Hyperpolarizing
from tests.helpers.os_utils import cwd


parent_dir = Path(__file__).resolve().parent.parent
cond_params_conf_path = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config.json"
)

multi_input_conf_path = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config_many_inputs.json"
)

hipp_sim_with_projection = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


def test_non_existing_config():
    with pytest.raises(FileNotFoundError):
        SonataSimulationConfig(parent_dir / "examples" / "non_existing_config")


def test_init_with_snap_simulation_config():
    snap_sim = SnapSimulation(cond_params_conf_path)
    SonataSimulationConfig(snap_sim)


def test_init_with_invalid_type():
    with pytest.raises(TypeError):
        SonataSimulationConfig({"invalid": "type", "dict": "config"})


def test_get_all_stimuli_entries():
    sim = SonataSimulationConfig(multi_input_conf_path)
    noise_stim = Noise("Mosaic_A", 10.0, 20.0, 200.0, 0.001)
    hyper_stim = Hyperpolarizing("Mosaic_A", 0.0, 50.0)
    entries = sim.get_all_stimuli_entries()
    assert len(entries) == 2
    assert entries[0] == noise_stim
    assert entries[1] == hyper_stim


def test_condition_parameters():
    sim = SonataSimulationConfig(cond_params_conf_path)
    conditions = sim.condition_parameters()
    assert conditions == Conditions(
        mech_conditions=MechanismConditions(
            ampanmda=ConditionEntry(minis_single_vesicle=0, init_depleted=1),
            gabaab=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
            glusynapse=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
        ),
        celsius=34.0,
        v_init=-80.0,
        extracellular_calcium=None,
        randomize_gaba_rise_time=False,
    )


def test_connection_entries():
    sim = SonataSimulationConfig(hipp_sim_with_projection)
    entries = sim.connection_entries()
    assert len(entries) == 4
    assert entries[0] == ConnectionOverrides(
        source="Mosaic",
        target="Mosaic",
        delay=None,
        weight=1.0,
        spont_minis=0.01,
        synapse_configure=None,
        mod_override=None,
    )
    assert entries[2].synapse_configure == "%s.NMDA_ratio = 1.22 %s.tau_r_NMDA = 3.9 %s.tau_d_NMDA = 148.5"
    assert entries[-1] == ConnectionOverrides(
        source="Excitatory",
        target="Mosaic",
        delay=None,
        weight=None,
        spont_minis=None,
        synapse_configure="%s.mg = 1.0",
        mod_override=None
    )


def test_connection_override():
    sim = SonataSimulationConfig(hipp_sim_with_projection)

    entries = sim.connection_entries()
    assert len(entries) == 4

    connection_override = ConnectionOverrides(
        source="Excitatory",
        target="Mosaic",
        delay=2,
        weight=2.0,
        spont_minis=0.1,
        synapse_configure="%s.mg = 1.4",
        mod_override=None
    )
    sim.add_connection_override(connection_override)

    entries = sim.connection_entries()
    assert len(entries) == 5
    assert entries[-1] == connection_override

    # overrides are not added multiple times
    entries = sim.connection_entries()
    entries = sim.connection_entries()
    assert len(entries) == 5


def test_get_all_projection_names():
    sim_dir = cond_params_conf_path.parent
    sim_config = cond_params_conf_path.name
    with cwd(sim_dir):
        sim = SonataSimulationConfig(sim_config)
        assert sim.get_all_projection_names() == []


def test_seeds():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.base_seed == 1
    assert sim.synapse_seed == 0
    assert sim.ionchannel_seed == 0
    assert sim.stimulus_seed == 0
    assert sim.minis_seed == 0


def test_rng_mode():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.rng_mode == "Random123"


def test_spike_threshold():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.spike_threshold == -30.0


def test_spike_location():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.spike_location == "soma"


def test_duration():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.duration == 50.0


def test_dt():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.dt == 0.025


def test_forward_skip():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.forward_skip is None


def test_celsius():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.celsius == 34.0


def test_v_init():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.v_init == -80.0


def test_output_root_path():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert Path(sim.output_root_path).name == "output_sonata"


def test_extracellular_calcium():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.extracellular_calcium is None
