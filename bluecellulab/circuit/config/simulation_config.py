# Copyright 2012-2023 Blue Brain Project / EPFL

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
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

from bluecellulab import BLUEPY_AVAILABLE
from bluecellulab.exceptions import ExtraDependencyMissingError

if BLUEPY_AVAILABLE:
    from bluepy_configfile.configfile import BlueConfig
    from bluepy.utils import open_utf8
from bluepysnap import Simulation as SnapSimulation

from bluecellulab.circuit.config.sections import Conditions, ConnectionOverrides
from bluecellulab.stimuli import Stimulus


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


class BluepySimulationConfig:
    """Bluepy implementation of SimulationConfig protocol."""
    _connection_overrides: list[ConnectionOverrides] = []

    def __init__(self, config: str | BlueConfig) -> None:
        if not BLUEPY_AVAILABLE:
            raise ExtraDependencyMissingError("bluepy")
        if isinstance(config, str):
            if not Path(config).exists():
                raise FileNotFoundError(f"Config file {config} not found.")
            else:
                with open_utf8(config) as f:
                    self.impl = BlueConfig(f)
        else:
            self.impl = config

    def get_all_projection_names(self) -> list[str]:
        unique_names = {proj.name for proj in self.impl.typed_sections('Projection')}
        return list(unique_names)

    def get_all_stimuli_entries(self) -> list[Stimulus]:
        """Get all stimuli entries."""
        result = []
        for entry in self.impl.typed_sections('StimulusInject'):
            # retrieve the stimulus to apply
            stimulus_name: str = entry.Stimulus
            # bluepy magic to add underscore Stimulus underscore
            # stimulus_name
            stimulus = self.impl['Stimulus_%s' % stimulus_name].to_dict()
            stimulus["Target"] = entry.Target
            stimulus = Stimulus.from_blueconfig(stimulus)
            if stimulus:
                result.append(stimulus)
        return result

    @lru_cache(maxsize=1)
    def condition_parameters(self) -> Conditions:
        """Returns parameters of global condition block of the blueconfig."""
        try:
            condition_entries = self.impl.typed_sections('Conditions')[0].to_dict()
        except IndexError:
            return Conditions.init_empty()
        return Conditions.from_blueconfig(condition_entries)

    @lru_cache(maxsize=1)
    def _connection_entries(self) -> list[ConnectionOverrides]:
        result = []
        for conn_entry in self.impl.typed_sections('Connection'):
            # SynapseID is ignored by bluepy thus not part of dict representation
            conn_entry_dict = conn_entry.to_dict()
            if "SynapseID" in conn_entry:
                conn_entry_dict["SynapseID"] = int(conn_entry["SynapseID"])
            result.append(ConnectionOverrides.from_blueconfig(conn_entry_dict))
        return result

    def connection_entries(self) -> list[ConnectionOverrides]:
        return self._connection_entries() + self._connection_overrides

    @property
    def base_seed(self) -> int:
        """Base seed of blueconfig."""
        return int(self.impl.Run['BaseSeed']) if 'BaseSeed' in self.impl.Run else 0

    @property
    def synapse_seed(self) -> int:
        """Synapse seed of blueconfig."""
        return int(self.impl.Run['SynapseSeed']) if 'SynapseSeed' in self.impl.Run else 0

    @property
    def ionchannel_seed(self) -> int:
        """Ion channel seed of blueconfig."""
        return int(self.impl.Run['IonChannelSeed']) if 'IonChannelSeed' in self.impl.Run else 0

    @property
    def stimulus_seed(self) -> int:
        """Stimulus seed of blueconfig."""
        return int(self.impl.Run['StimulusSeed']) if 'StimulusSeed' in self.impl.Run else 0

    @property
    def minis_seed(self) -> int:
        """Minis seed of blueconfig."""
        return int(self.impl.Run['MinisSeed']) if 'MinisSeed' in self.impl.Run else 0

    @property
    def rng_mode(self) -> str:
        """Gets the rng mode defined in simulation."""
        # Ugly, but mimicking neurodamus
        if 'Simulator' in self.impl.Run and self.impl.Run['Simulator'] != 'NEURON':
            return 'Random123'
        elif 'RNGMode' in self.impl.Run:
            return self.impl.Run['RNGMode']
        else:
            return "Compatibility"  # default rng mode

    @property
    def spike_threshold(self) -> float:
        """Get the spike threshold from simulation config."""
        if 'SpikeThreshold' in self.impl.Run:
            return float(self.impl.Run['SpikeThreshold'])
        else:
            return -30.0

    @property
    def spike_location(self) -> str:
        """Get the spike location from simulation config."""
        if 'SpikeLocation' in self.impl.Run:
            return self.impl.Run['SpikeLocation']
        else:
            return "soma"

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the simulation."""
        if 'Duration' in self.impl.Run:
            return float(self.impl.Run['Duration'])
        else:
            return None

    @property
    def dt(self) -> float:
        return float(self.impl.Run['Dt'])

    @property
    def _soma_report_dt(self) -> float:
        return float(self.impl.Report_soma['Dt'])

    @property
    def forward_skip(self) -> Optional[float]:
        if 'ForwardSkip' in self.impl.Run:
            return float(self.impl.Run['ForwardSkip'])
        return None

    @property
    def celsius(self) -> float:
        if 'Celsius' in self.impl.Run:
            return float(self.impl.Run['Celsius'])
        else:
            return 34.0  # default

    @property
    def v_init(self) -> float:
        if 'V_Init' in self.impl.Run:
            return float(self.impl.Run['V_Init'])
        else:
            return -65.0

    @property
    def output_root_path(self) -> str:
        """Get the output root path."""
        return self.impl.Run['OutputRoot']

    @property
    def extracellular_calcium(self) -> Optional[float]:
        """Get the extracellular calcium value."""
        if 'ExtracellularCalcium' in self.impl.Run:
            return float(self.impl.Run['ExtracellularCalcium'])
        else:
            return None

    def add_connection_override(
        self,
        connection_override: ConnectionOverrides
    ) -> None:
        self._connection_overrides.append(connection_override)


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
