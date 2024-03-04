# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

from bluecellulab.circuit.config.sections import Conditions, ConnectionOverrides
from bluecellulab.stimulus.circuit_stimulus_definitions import Stimulus

from bluepysnap import Simulation as SnapSimulation


class SonataSimulationConfig:
    """Sonata implementation of SimulationConfig protocol."""
    _connection_overrides: list[ConnectionOverrides] = []

    def __init__(self, config: str | Path | SnapSimulation) -> None:
        if isinstance(config, (str, Path)):
            if not Path(config).exists():
                raise FileNotFoundError(f"Config file {config} not found.")
            else:
                self.impl = SnapSimulation(config)
        elif isinstance(config, SnapSimulation):
            self.impl = config
        else:
            raise TypeError("Invalid config type.")

    def get_all_projection_names(self) -> list[str]:
        unique_names = {
            n
            for n in self.impl.circuit.nodes
            if self.impl.circuit.nodes[n].type == "virtual"
        }
        return list(unique_names)

    def get_all_stimuli_entries(self) -> list[Stimulus]:
        result: list[Stimulus] = []
        inputs = self.impl.config.get("inputs")
        if inputs is None:
            return result
        for value in inputs.values():
            stimulus = Stimulus.from_sonata(value)
            if stimulus:
                result.append(stimulus)
        return result

    @lru_cache(maxsize=1)
    def condition_parameters(self) -> Conditions:
        """Returns parameters of global condition block of sonataconfig."""
        condition_object = self.impl.conditions
        return Conditions.from_sonata(condition_object)

    @lru_cache(maxsize=1)
    def _connection_entries(self) -> list[ConnectionOverrides]:
        result: list[ConnectionOverrides] = []
        if "connection_overrides" not in self.impl.config:
            return result
        conn_overrides: list = self.impl.config["connection_overrides"]
        if conn_overrides is None:
            return result
        for conn_entry in conn_overrides:
            result.append(ConnectionOverrides.from_sonata(conn_entry))
        return result

    def connection_entries(self) -> list[ConnectionOverrides]:
        return self._connection_entries() + self._connection_overrides

    @property
    def base_seed(self) -> int:
        return self.impl.run.random_seed

    @property
    def synapse_seed(self) -> int:
        return self.impl.run.synapse_seed

    @property
    def ionchannel_seed(self) -> int:
        return self.impl.run.ionchannel_seed

    @property
    def stimulus_seed(self) -> int:
        return self.impl.run.stimulus_seed

    @property
    def minis_seed(self) -> int:
        return self.impl.run.minis_seed

    @property
    def rng_mode(self) -> str:
        """Only Random123 is supported in SONATA."""
        return "Random123"

    @property
    def spike_threshold(self) -> float:
        return self.impl.run.spike_threshold

    @property
    def spike_location(self) -> str:
        return self.impl.conditions.spike_location.name

    @property
    def duration(self) -> Optional[float]:
        return self.impl.run.tstop

    @property
    def dt(self) -> float:
        return self.impl.run.dt

    @property
    def forward_skip(self) -> Optional[float]:
        """forward_skip is removed from SONATA."""
        return None

    @property
    def celsius(self) -> float:
        value = self.condition_parameters().celsius
        return value if value is not None else 34.0

    @property
    def v_init(self) -> float:
        value = self.condition_parameters().v_init
        return value if value is not None else -65.0

    @property
    def output_root_path(self) -> str:
        return self.impl.config["output"]["output_dir"]

    @property
    def extracellular_calcium(self) -> Optional[float]:
        return self.condition_parameters().extracellular_calcium

    def add_connection_override(
        self,
        connection_override: ConnectionOverrides
    ) -> None:
        self._connection_overrides.append(connection_override)
