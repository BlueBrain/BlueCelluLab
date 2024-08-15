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
"""Unit tests for the stimuli module."""

from pydantic import ValidationError
import pytest
from bluecellulab.stimulus.circuit_stimulus_definitions import SynapseReplay


def test_synapse_replay_validator():
    """Assures synapse replay's validator fails."""
    with pytest.raises(ValidationError):
        _ = SynapseReplay(
            target="target1",
            delay=0,
            duration=3000,
            spike_file="file_that_does_not_exist.dat",
        )
