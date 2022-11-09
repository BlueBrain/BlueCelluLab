"""Layer to abstract the circuit access functionality from the rest of BGLibPy."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import bluepy
from bluepy_configfile.configfile import BlueConfig
from bluepy.enums import Cell
from cachetools import LRUCache, cachedmethod
import numpy as np
import pandas as pd

from bglibpy.exceptions import BGLibPyError
from bglibpy.circuit.config import SimulationConfig


class CircuitAccess:
    """Encapsulated circuit data and functions."""

    def __init__(self, circuit_config: str | BlueConfig) -> None:
        """Initialize bluepy circuit object."""
        if isinstance(circuit_config, str) and not Path(circuit_config).exists():
            raise FileNotFoundError(f"Circuit config file {circuit_config} not found.")

        self._bluepy_sim = bluepy.Simulation(circuit_config)
        self._bluepy_circuit = self._bluepy_sim.circuit
        self.config = SimulationConfig(self._bluepy_sim.config)

        self._caches: dict = {
            "is_group_target": LRUCache(maxsize=1000),
            "target_has_gid": LRUCache(maxsize=1000),
            "_get_target_gids": LRUCache(maxsize=16),
            "fetch_gid_cell_info": LRUCache(maxsize=100),
            "condition_parameters_dict": LRUCache(maxsize=1)
        }

    @property
    def available_cell_properties(self) -> set:
        """Retrieve the available properties from connectome."""
        return self._bluepy_circuit.cells.available_properties

    def get_emodel_info(self, gid: int) -> dict:
        """Return the emodel info for a gid."""
        return self._bluepy_circuit.emodels.get_mecombo_info(gid)

    def get_cell_ids(self, group: str) -> set[int]:
        """Return the cell ids of a group."""
        return set(self._bluepy_circuit.cells.ids(group))

    def get_cell_properties(
        self, gid: int, properties: list[str] | str
    ) -> pd.DataFrame | pd.Series:
        """Get a property of a cell."""
        if isinstance(properties, str):
            properties = [properties]
        return self._bluepy_circuit.cells.get(gid, properties=properties)

    def get_connectomes_dict(self, projections: Optional[list[str] | str]) -> dict:
        """Get the connectomes dictionary indexed by projections or connectome ids."""
        if isinstance(projections, str):
            projections = [projections]

        connectomes = {'': self._bluepy_circuit.connectome}
        if projections is not None:
            proj_conns = {p: self._bluepy_circuit.projection(p) for p in projections}
            connectomes.update(proj_conns)

        return connectomes

    def get_soma_voltage(
        self, gid: int, t_start: float, t_end: float, t_step: float
    ) -> np.ndarray:
        """Retrieve the soma voltage of main simulation."""
        return (
            self._bluepy_sim.report("soma")
            .get_gid(gid, t_start=t_start, t_end=t_end, t_step=t_step)
            .to_numpy()
        )

    def get_soma_time_trace(self) -> np.ndarray:
        """Retrieve the time trace from the main simulation."""
        report = self._bluepy_sim.report('soma')
        return report.get_gid(report.gids[0]).index.to_numpy()

    @property
    def use_mecombo_tsv(self) -> bool:
        """Property that decides whether to use mecombo_tsv."""
        _use_mecombo_tsv = False
        if self.node_properties_available:
            _use_mecombo_tsv = False
        elif 'MEComboInfoFile' in self.config.bc.Run:
            _use_mecombo_tsv = True
        else:
            _use_mecombo_tsv = False
        return _use_mecombo_tsv

    @staticmethod
    def is_cell_target(target, gid: int) -> bool:
        """Check if target is a cell."""
        return target == 'a%d' % gid

    @cachedmethod(lambda self: self._caches["is_group_target"])
    def is_group_target(self, target: str) -> bool:
        """Check if target is a group of cells."""
        return target in self._bluepy_circuit.cells.targets

    @cachedmethod(lambda self: self._caches["_get_target_gids"])
    def _get_target_gids(self, target: str) -> set:
        """Return GIDs in target as a set."""
        return set(self._bluepy_circuit.cells.ids(target))

    @cachedmethod(lambda self: self._caches["target_has_gid"])
    def target_has_gid(self, target: str, gid: int) -> bool:
        """Checks if target has the gid."""
        return gid in self._get_target_gids(target)

    @cachedmethod(lambda self: self._caches["fetch_gid_cell_info"])
    def fetch_gid_cell_info(self, gid: int) -> pd.DataFrame:
        """Fetch bluepy cell info of a gid"""
        if gid in self._bluepy_circuit.cells.ids():
            return self._bluepy_circuit.cells.get(gid)
        else:
            raise BGLibPyError(f"Gid {gid} not found in circuit")

    def fetch_mecombo_name(self, gid: int) -> str:
        """Fetch mecombo name for a certain gid."""
        cell_info = self.fetch_gid_cell_info(gid)
        if self.node_properties_available:
            me_combo = str(cell_info['model_template'])
            me_combo = me_combo.split('hoc:')[1]
        else:
            me_combo = str(cell_info['me_combo'])
        return me_combo

    def fetch_emodel_name(self, gid: int) -> str:
        """Get the emodel path of a gid."""
        me_combo = self.fetch_mecombo_name(gid)
        if self.use_mecombo_tsv:
            emodel_name = self._bluepy_circuit.emodels.get_mecombo_info(gid)["emodel"]
        else:
            emodel_name = me_combo

        return emodel_name

    def fetch_morph_name(self, gid: int) -> str:
        """Get the morph name of a gid."""
        cell_info = self.fetch_gid_cell_info(gid)
        return str(cell_info['morphology'])

    def fetch_mini_frequencies(self, gid: int) -> tuple:
        """Get inhibitory frequency of gid."""
        cell_info = self.fetch_gid_cell_info(gid)

        inh_mini_frequency = cell_info['inh_mini_frequency'] \
            if 'inh_mini_frequency' in cell_info else None
        exc_mini_frequency = cell_info['exc_mini_frequency'] \
            if 'exc_mini_frequency' in cell_info else None

        return exc_mini_frequency, inh_mini_frequency

    @property
    def node_properties_available(self) -> bool:
        """Checks if the node properties are available and can be used."""
        node_props = {
            "@dynamics:holding_current",
            "@dynamics:threshold_current",
            "model_template",
        }

        return node_props.issubset(self._bluepy_circuit.cells.available_properties)

    def get_gids_of_mtypes(self, mtypes: list[str]) -> set[int]:
        """Returns all the gids belonging to one of the input mtypes."""
        gids = set()
        for mtype in mtypes:
            gids |= set(self._bluepy_circuit.cells.ids({Cell.MTYPE: mtype}))

        return gids

    def get_gids_of_targets(self, targets: list[str]) -> set[int]:
        """Return all the gids belonging to one of the input targets."""
        gids = set()
        for target in targets:
            gids |= set(self._bluepy_circuit.cells.ids(target))

        return gids

    def morph_filename(self, gid: int) -> str:
        return f"{self.fetch_morph_name(gid)}.{self.config.morph_extension}"

    def emodel_path(self, gid: int) -> str:
        return os.path.join(self.config.emodels_dir, f"{self.fetch_emodel_name(gid)}.hoc")
