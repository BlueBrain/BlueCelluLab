"""Testing SSim with SONATA simulations."""

from pathlib import Path

import numpy as np
import pytest

from bluecellulab import SSim

parent_dir = Path(__file__).resolve().parent


INPUTS_TO_TEST = ["noinput", "hypamp", "ornstein", "shotnoise"]


@pytest.mark.v6
@pytest.mark.parametrize("input_type", ["noinput", "hypamp"])
def test_sim_quick_scx_sonata(input_type):
    """Test against sim results of quick_scx_sonata."""
    # Path to the SONATA simulation
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sim_quick_scx_sonata"
        / f"simulation_config_{input_type}.json"
    )

    # Create SSim object
    sim = SSim(sonata_sim_path)
    cell_id = ("NodeA", 2)  # has a spike + hyperpolarises
    sim.instantiate_gids(cell_id, add_stimuli=True)
    t_stop = 10.0
    sim.run(t_stop)

    # Get the voltage trace
    time = sim.get_time_trace(1)
    voltage = sim.get_voltage_trace(cell_id, 0, t_stop, 1)
    voltage = voltage[:len(voltage) - 1]  # remove last point, mainsim produces 1 less
    time = time[:len(time) - 1]  # remove last point, mainsim produces 1 less
    mainsim_voltage = sim.get_mainsim_voltage_trace(cell_id)
    voltage_diff = voltage - mainsim_voltage
    rms_error = np.sqrt(np.mean(voltage_diff ** 2))
    assert rms_error < 1e-4


@pytest.mark.v6
@pytest.mark.parametrize("input_type", INPUTS_TO_TEST)
def test_sim_quick_scx_sonata_multicircuit(input_type):
    """Sonata config multicircuit test.
       Applies the stimulus defined in INPUTS_TO_TEST one by one.
    """
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sim_quick_scx_sonata_multicircuit"
        / f"simulation_config_{input_type}.json"
    )

    cell_ids = [("NodeA", 1), ("NodeA", 2)]
    # investivate NodeA, 0 further. It shows small discrepancies,
    # even on a single population circuit
    # Create SSim object
    sim = SSim(sonata_sim_path)
    sim.instantiate_gids(cell_ids, add_stimuli=True)
    t_stop = 20.0
    sim.run(t_stop)
    for cell_id in cell_ids:
        voltage = sim.get_voltage_trace(cell_id, 0, t_stop, 1)
        voltage = voltage[:len(voltage) - 1]  # remove last point, mainsim produces 1 less
        mainsim_voltage = sim.get_mainsim_voltage_trace(cell_id)
        voltage_diff = voltage - mainsim_voltage
        rms_error = np.sqrt(np.mean(voltage_diff ** 2))
        assert rms_error < 0.25


@pytest.mark.v6
def test_ssim_intersect_pre_gids_multipopulation():
    """Test instantiate_gids with intersect_pre_gids on Sonata."""
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sim_quick_scx_sonata_multicircuit"
        / "simulation_config_noinput.json"
    )
    cell_ids = [("NodeA", 0), ("NodeA", 1)]

    sim = SSim(sonata_sim_path)
    sim.instantiate_gids(cell_ids, add_synapses=True)
    assert len([x.synapses for x in sim.cells.values()][0]) == 6
    assert len([x.synapses for x in sim.cells.values()][1]) == 2

    # pre gids are intersected, synapses are filtered
    sim2 = SSim(sonata_sim_path)
    sim2.instantiate_gids(cell_ids, add_synapses=True, intersect_pre_gids=[("NodeB", 0)])
    assert len([x.synapses for x in sim2.cells.values()][0]) == 2
    assert len([x.synapses for x in sim2.cells.values()][1]) == 0


@pytest.mark.v6
def test_merge_pre_spike_trains_edge_case():
    """Test to make sure connections get initialised.
   for fixing empty cell_info_dict['connections'] bug."""
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sim_quick_scx_sonata_multicircuit"
        / "simulation_config_noinput.json"
    )
    cell_id = ("NodeA", 0)
    ssim = SSim(sonata_sim_path)
    ssim.instantiate_gids(cell_id, add_minis=True, add_replay=True,
                          add_stimuli=False, add_synapses=True,
                          intersect_pre_gids=None)
    cell_info_dict = ssim.cells[cell_id].info_dict
    assert cell_info_dict["connections"] != {}
