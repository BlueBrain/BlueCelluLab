"""Unit tests for the simulation_access module."""

from pathlib import Path

import numpy as np
import pytest

from bluecellulab.circuit import CellId, SonataSimulationAccess
from bluecellulab.circuit.simulation_access import _sample_array, get_synapse_replay_spikes


parent_dir = Path(__file__).resolve().parent.parent

hipp_simulation_with_projections = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


def test_sample_array():
    """Unit test for _sample_array."""
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    t_step = 0.1
    sim_t_step = 0.1
    assert np.array_equal(_sample_array(arr, t_step / sim_t_step), arr)
    t_step = 0.2
    sim_t_step = 0.1
    assert np.array_equal(_sample_array(arr, t_step / sim_t_step), [1, 3, 5, 7, 9])
    t_step = 0.3
    sim_t_step = 0.1
    assert np.array_equal(_sample_array(arr, t_step / sim_t_step), [1, 4, 7, 10])
    t_step = 0.4
    sim_t_step = 0.1
    assert np.array_equal(_sample_array(arr, t_step / sim_t_step), [1, 5, 9])
    t_step = 0.1
    sim_t_step = 0.2
    with pytest.raises(ValueError):
        _sample_array(arr, t_step / sim_t_step)


class TestSonataSimulationAccess:

    def setup(self):
        self.simulation_access = SonataSimulationAccess(str(hipp_simulation_with_projections))

    @staticmethod
    def test_init_file_not_found():
        """Test BluepySimulationAccess init edge cases."""
        with pytest.raises(FileNotFoundError):
            SonataSimulationAccess(parent_dir / "examples" / "non_existing_file")

    def test_get_soma_voltage(self):
        """Test SonataCircuitAccess.get_soma_voltage."""
        cell_id = CellId("hippocampus_neurons", 1)
        t_start, t_end, t_step, sim_t_step = 0, 100, 0.1, 0.025
        soma_voltage = self.simulation_access.get_soma_voltage(cell_id, t_start, t_end, t_step)
        t_step_ratio = round(t_step / sim_t_step)
        assert soma_voltage.shape == (int((t_end - t_start) / t_step / t_step_ratio),)
        assert soma_voltage[0] == -65.0
        assert np.mean(soma_voltage) == pytest.approx(-54.447914)
        assert np.max(soma_voltage) == pytest.approx(41.53086)
        assert np.min(soma_voltage) == pytest.approx(-77.127716)

    def test_get_soma_time_trace(self):
        """Test SonataCircuitAccess.get_soma_time_trace."""
        t_step = 0.1
        soma_time_trace = self.simulation_access.get_soma_time_trace(t_step)
        assert soma_time_trace.shape == (250,)
        assert soma_time_trace[0] == 0.0
        assert soma_time_trace[1] == 0.4
        assert soma_time_trace[-1] == pytest.approx(99.6)

    def test_get_spikes(self):
        """Test SonataCircuitAccess.get_spikes."""
        spikes = self.simulation_access.get_spikes()
        assert len(spikes.keys()) == 10
        cell_id = CellId("hippocampus_neurons", 2)
        assert len(spikes[cell_id]) == 4
        np.testing.assert_almost_equal(
            spikes[cell_id], np.array([10.1, 22.65, 35.15, 47.675]))


def test_get_synapse_replay_spikes():
    """.Test get_synapse_replay_spikes."""
    res = get_synapse_replay_spikes(
        parent_dir / "data" / "synapse_replay_file" / "spikes.dat"
    )
    assert set(res.keys()) == {5382}
    assert res[5382].tolist() == [1500.0, 2000.0, 2500.0]
