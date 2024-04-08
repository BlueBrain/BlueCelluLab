"""Unit tests for CircuitSimulation."""
import numpy as np

from bluecellulab import CircuitSimulation


def test_merge_pre_spike_trains():
    """CircuitSimulation: Testing merge_pre_spike_trains"""

    train1 = {1: [5, 100], 2: [5, 8, 120]}
    train2 = {2: [7], 3: [8]}
    train3 = {1: [5, 100]}

    trains_merged = {1: [5, 5, 100, 100], 2: [5, 7, 8, 120], 3: [8]}

    np.testing.assert_equal(
        {},
        CircuitSimulation.merge_pre_spike_trains(None))
    np.testing.assert_equal(
        train1,
        CircuitSimulation.merge_pre_spike_trains(train1))
    np.testing.assert_equal(
        train1,
        CircuitSimulation.merge_pre_spike_trains(
            None,
            train1))
    np.testing.assert_equal(
        trains_merged,
        CircuitSimulation.merge_pre_spike_trains(
            train1,
            None,
            train2,
            train3))
