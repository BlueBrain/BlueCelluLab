# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
