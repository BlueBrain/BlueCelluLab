"""Layer to abstract the circuit access functionality from the rest of BGLibPy."""

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import bluepy
from bluepy_configfile.configfile import BlueConfig
from bluepy.enums import Cell
from cachetools import LRUCache, cachedmethod
import pandas as pd

from bglibpy.exceptions import BGLibPyError


class CircuitAccess:
    """Encapsulated circuit data and functions."""

    def __init__(self, circuit_config: Union[str, BlueConfig]) -> None:
        """Initialize bluepy circuit object."""
        if isinstance(circuit_config, str) and not Path(circuit_config).exists():
            raise FileNotFoundError(f"Circuit config file {circuit_config} not found.")

        self.bluepy_sim = bluepy.Simulation(circuit_config)
        self.bluepy_circuit = self.bluepy_sim.circuit
        self.bc = self.bluepy_sim.config

        self._caches: dict = {
            "is_group_target": LRUCache(maxsize=1000),
            "target_has_gid": LRUCache(maxsize=1000),
            "_get_target_gids": LRUCache(maxsize=16),
            "fetch_gid_cell_info": LRUCache(maxsize=100),
            "condition_parameters_dict": LRUCache(maxsize=1)
        }

    @property
    def use_mecombo_tsv(self) -> bool:
        """Property that decides whether to use mecombo_tsv."""
        _use_mecombo_tsv = False
        if self.node_properties_available:
            _use_mecombo_tsv = False
        elif 'MEComboInfoFile' in self.bc.Run:
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
        return target in self.bluepy_circuit.cells.targets

    @cachedmethod(lambda self: self._caches["_get_target_gids"])
    def _get_target_gids(self, target: str) -> set:
        """Return GIDs in target as a set."""
        return set(self.bluepy_circuit.cells.ids(target))

    @cachedmethod(lambda self: self._caches["target_has_gid"])
    def target_has_gid(self, target: str, gid: int) -> bool:
        """Checks if target has the gid."""
        return gid in self._get_target_gids(target)

    @cachedmethod(lambda self: self._caches["fetch_gid_cell_info"])
    def fetch_gid_cell_info(self, gid: int) -> pd.DataFrame:
        """Fetch bluepy cell info of a gid"""
        if gid in self.bluepy_circuit.cells.ids():
            return self.bluepy_circuit.cells.get(gid)
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
            emodel_name = self.bluepy_circuit.emodels.get_mecombo_info(gid)["emodel"]
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

        return node_props.issubset(self.bluepy_circuit.cells.available_properties)

    @cachedmethod(lambda self: self._caches["condition_parameters_dict"])
    def condition_parameters_dict(self) -> dict:
        """Returns parameters of global condition block of the blueconfig."""
        try:
            condition_entries = self.bc.typed_sections('Conditions')[0]
        except IndexError:
            return {}
        return condition_entries.to_dict()

    @property
    def connection_entries(self):
        return self.bc.typed_sections('Connection')

    @property
    def is_glusynapse_used(self) -> bool:
        """Checks if glusynapse is used in the simulation config."""
        is_glusynapse_used = any(
            x
            for x in self.connection_entries
            if "ModOverride" in x and x["ModOverride"] == "GluSynapse"
        )
        return is_glusynapse_used

    def get_gids_of_mtypes(self, mtypes: List[str]) -> Set[int]:
        """Returns all the gids belonging to one of the input mtypes."""
        gids = set()
        for mtype in mtypes:
            gids |= set(self.bluepy_circuit.cells.ids({Cell.MTYPE: mtype}))

        return gids

    def get_gids_of_targets(self, targets: List[str]) -> Set[int]:
        """Return all the gids belonging to one of the input targets."""
        gids = set()
        for target in targets:
            gids |= set(self.bluepy_circuit.cells.ids(target))

        return gids

    def get_morph_dir_and_extension(self) -> Tuple[str, str]:
        """Get the tuple of morph_dir and extension."""
        morph_dir = self.bc.Run['MorphologyPath']

        if 'MorphologyType' in self.bc.Run:
            morph_extension = self.bc.Run['MorphologyType']
        else:
            # backwards compatible
            if morph_dir[-3:] == "/h5":
                morph_dir = morph_dir[:-3]

            # latest circuits don't have asc dir
            asc_dir = os.path.join(morph_dir, 'ascii')
            if os.path.exists(asc_dir):
                morph_dir = asc_dir

            morph_extension = 'asc'

        return morph_dir, morph_extension

    @property
    def morph_dir(self) -> str:
        _morph_dir, _ = self.get_morph_dir_and_extension()
        return _morph_dir

    @property
    def morph_extension(self) -> str:
        _, _morph_extension = self.get_morph_dir_and_extension()
        return _morph_extension

    def morph_filename(self, gid: int) -> str:
        return f"{self.fetch_morph_name(gid)}.{self.morph_extension}"

    @property
    def emodels_dir(self) -> str:
        return self.bc.Run['METypePath']

    def emodel_path(self, gid: int) -> str:
        return os.path.join(self.emodels_dir, f"{self.fetch_emodel_name(gid)}.hoc")

    @property
    def extracellular_calcium(self) -> Optional[float]:
        """Get the extracellular calcium value."""
        if 'ExtracellularCalcium' in self.bc.Run:
            return float(self.bc.Run['ExtracellularCalcium'])
        else:
            return None
