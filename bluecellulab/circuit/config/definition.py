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
"""Module that interacts with the simulation config."""

from __future__ import annotations
from typing import Optional, Protocol


from bluecellulab.circuit.config.sections import Conditions, ConnectionOverrides
from bluecellulab.stimulus.circuit_stimulus_definitions import Stimulus


class SimulationConfig(Protocol):
    """Protocol that defines the simulation config layer."""

    def get_all_stimuli_entries(self) -> list[Stimulus]:
        raise NotImplementedError

    def get_all_projection_names(self) -> list[str]:
        raise NotImplementedError

    def condition_parameters(self) -> Conditions:
        raise NotImplementedError

    def connection_entries(self) -> list[ConnectionOverrides]:
        raise NotImplementedError

    @property
    def base_seed(self) -> int:
        raise NotImplementedError

    @property
    def synapse_seed(self) -> int:
        raise NotImplementedError

    @property
    def ionchannel_seed(self) -> int:
        raise NotImplementedError

    @property
    def stimulus_seed(self) -> int:
        raise NotImplementedError

    @property
    def minis_seed(self) -> int:
        raise NotImplementedError

    @property
    def rng_mode(self) -> str:
        raise NotImplementedError

    @property
    def spike_threshold(self) -> float:
        raise NotImplementedError

    @property
    def spike_location(self) -> str:
        raise NotImplementedError

    @property
    def duration(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def dt(self) -> float:
        raise NotImplementedError

    @property
    def forward_skip(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def celsius(self) -> float:
        raise NotImplementedError

    @property
    def v_init(self) -> float:
        raise NotImplementedError

    @property
    def output_root_path(self) -> str:
        raise NotImplementedError

    @property
    def extracellular_calcium(self) -> Optional[float]:
        raise NotImplementedError

    def add_connection_override(
        self,
        connection_override: ConnectionOverrides
    ) -> None:
        raise NotImplementedError
