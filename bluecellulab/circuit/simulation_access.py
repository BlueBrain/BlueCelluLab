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
"""Module to access the simulation results."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Protocol

import h5py
import pandas as pd

from bluecellulab.circuit.config import SimulationConfig, SonataSimulationConfig
from bluecellulab.exceptions import ExtraDependencyMissingError

from bluecellulab import BLUEPY_AVAILABLE
if BLUEPY_AVAILABLE:
    import bluepy
    from bluecellulab.circuit.iotools import parse_outdat

from bluepysnap import Simulation as SnapSimulation
import numpy as np

from bluecellulab.circuit import CellId
from bluecellulab.circuit.config import BluepySimulationConfig


logger = logging.getLogger(__name__)


def _sample_array(arr: np.ndarray, ratio: float) -> np.ndarray:
    """Sample an array at a given time step.

    Args:
        arr: Array to sample.
        ratio: The ratio between the time step used for sampling and the time step used in the simulation
                (the time step should be a multiple of the simulation time step)

    Returns:
        Array sampled at the given time step.
    """
    if ratio == 1:
        return arr
    elif not np.isclose(ratio, round(ratio)):
        raise ValueError("The ratio is not close to a whole number. "
                         "The time step should be a multiple of the simulation time step.")
    return arr[::round(ratio)]


class SimulationAccess(Protocol):
    """Protocol that defines the simulation access layer."""

    impl: bluepy.Simulation | SnapSimulation

    def get_soma_voltage(
        self, cell_id: CellId, t_start: float, t_end: float, t_step: Optional[float] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def get_soma_time_trace(self, t_step: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def get_spikes(self) -> dict[CellId, np.ndarray]:
        """Get spikes from the main simulation."""
        raise NotImplementedError


class BluepySimulationAccess:
    """Bluepy implementation of SimulationAccess protocol."""

    def __init__(self, sim_config: str | Path | SimulationConfig) -> None:
        """Initialize the simulation access object."""
        if not BLUEPY_AVAILABLE:
            raise ExtraDependencyMissingError("bluepy")
        if isinstance(sim_config, BluepySimulationConfig):
            sim_config = sim_config.impl
        elif isinstance(sim_config, Path):
            sim_config = str(sim_config)
        if isinstance(sim_config, str) and not Path(sim_config).exists():
            raise FileNotFoundError(
                f"Circuit config file {sim_config} not found.")

        self.impl = bluepy.Simulation(sim_config)
        self._config = BluepySimulationConfig(sim_config)  # type: ignore

    def get_soma_voltage(
        self, cell_id: CellId, t_start: float, t_end: float, t_step: Optional[float] = None
    ) -> np.ndarray:
        """Retrieve the soma voltage of main simulation."""
        gid = cell_id.id
        arr = (
            self.impl.report("soma")
            .get_gid(gid, t_start=t_start, t_end=t_end)
            .to_numpy()
        )
        if t_step is not None:
            ratio = t_step / self._config._soma_report_dt
            arr = _sample_array(arr, ratio)
        return arr

    def get_soma_time_trace(self, t_step: Optional[float] = None) -> np.ndarray:
        """Retrieve the time trace from the main simulation."""
        report = self.impl.report('soma')
        arr = report.get_gid(report.gids[0]).index.to_numpy()
        if t_step is not None:
            ratio = t_step / self._config._soma_report_dt
            arr = _sample_array(arr, ratio)
        return arr

    def get_spikes(self) -> dict[CellId, np.ndarray]:
        outdat_path = Path(self._config.output_root_path) / "out.dat"
        return parse_outdat(outdat_path)


class SonataSimulationAccess:
    """Sonata implementation of SimulationAccess protocol."""

    def __init__(self, sim_config: str | Path | SimulationConfig) -> None:
        """Initialize SonataCircuitAccess object."""
        if isinstance(sim_config, (str, Path)) and not Path(sim_config).exists():
            raise FileNotFoundError(f"Circuit config file {sim_config} not found.")

        if isinstance(sim_config, SonataSimulationConfig):
            self.impl = sim_config.impl
        else:
            self.impl = SnapSimulation(sim_config)

    def get_soma_voltage(
        self, cell_id: CellId, t_start: float, t_end: float, t_step: Optional[float] = None
    ) -> np.ndarray:
        report = self.impl.reports["soma"].filter(cell_id.id, t_start, t_end)
        arr = report.report[cell_id.population_name][cell_id.id].values
        if t_step is not None:
            ratio = t_step / self.impl.dt
            arr = _sample_array(arr, ratio)
        return arr

    def get_soma_time_trace(self, t_step: Optional[float] = None) -> np.ndarray:
        report = self.impl.reports["soma"]
        arr = report.filter().report.index.values
        if t_step is not None:
            ratio = t_step / self.impl.dt
            arr = _sample_array(arr, ratio)
        return arr

    def get_spikes(self) -> dict[CellId, np.ndarray]:
        spike_report = self.impl.spikes
        filtered_report = spike_report.filter()
        # filtered_report.head(2)
        #         ids population
        # times
        # 10.1     0  hippocampus_neurons
        # 10.1     1  hippocampus_neurons

        # convert it to dict, where key is CellId made of (population and ids columns)
        # and value is np.array of spike times
        outdat = filtered_report.report.groupby(["population", "ids"])
        outdat = outdat.apply(lambda x: x.index.values)
        outdat.index = [CellId(pop_name, idx) for (pop_name, idx) in outdat.index]
        return outdat.to_dict()


def get_synapse_replay_spikes(f_name: str) -> dict:
    """Read the .h5 file containing the spike replays."""
    with h5py.File(f_name, 'r') as f:
        # Access the timestamps and node_ids datasets
        timestamps = f['/spikes/All/timestamps'][:]
        node_ids = f['/spikes/All/node_ids'][:]

        spikes = pd.DataFrame(data={'t': timestamps, 'node_id': node_ids})
        spikes = spikes.astype({"node_id": int})

    if (spikes["t"] < 0).any():
        logger.warning("Found negative spike times... Clipping them to 0")
        spikes["t"].clip(lower=0., inplace=True)
    return spikes.groupby("node_id")["t"].apply(np.array).to_dict()
