"""Module to access the simulation results."""

from __future__ import annotations
from pathlib import Path
from platform import python_version_tuple
from typing import Optional, Protocol

from bglibpy.circuit.config import SimulationConfig, SonataSimulationConfig

if python_version_tuple() < ('3', '9'):
    from typing import Sequence
else:
    from collections.abc import Sequence

import bluepy
from bluepysnap import Simulation as SnapSimulation
import numpy as np

from bglibpy.circuit import CellId
from bglibpy.circuit.config import BluepySimulationConfig
from bglibpy.circuit.iotools import parse_outdat


def _sample_array(arr: Sequence, t_step: float, sim_t_step: float) -> Sequence:
    """Sample an array at a given time step.

    Args:
        arr: Array to sample.
        t_step: User specified time step to sample at.
        sim_t_step: Time step used in the main simulation.

    Returns:
        Array sampled at the given time step.
    """
    ratio = t_step / sim_t_step
    if t_step == sim_t_step:
        return arr
    elif not np.isclose(ratio, round(ratio)):
        raise ValueError(
            f"Time step {t_step} is not a multiple of the simulation time step {sim_t_step}.")
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
        if isinstance(sim_config, BluepySimulationConfig):
            sim_config = sim_config.impl
        elif isinstance(sim_config, Path):
            sim_config = str(sim_config)
        if isinstance(sim_config, str) and not Path(sim_config).exists():
            raise FileNotFoundError(
                f"Circuit config file {sim_config} not found.")

        self.impl = bluepy.Simulation(sim_config)
        self._config = BluepySimulationConfig(sim_config)

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
            arr = _sample_array(arr, t_step, self._config._soma_report_dt)
        return arr

    def get_soma_time_trace(self, t_step: Optional[float] = None) -> np.ndarray:
        """Retrieve the time trace from the main simulation."""
        report = self.impl.report('soma')
        arr = report.get_gid(report.gids[0]).index.to_numpy()
        if t_step is not None:
            arr = _sample_array(arr, t_step, self._config._soma_report_dt)
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
            arr = _sample_array(arr, t_step, self.impl.dt)
        return arr

    def get_soma_time_trace(self, t_step: Optional[float] = None) -> np.ndarray:
        report = self.impl.reports["soma"]
        arr = report.filter().report.index.values
        if t_step is not None:
            arr = _sample_array(arr, t_step, self.impl.dt)
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
