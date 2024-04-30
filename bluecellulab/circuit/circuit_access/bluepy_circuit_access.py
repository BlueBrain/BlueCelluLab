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
import logging
import os
from pathlib import Path
from typing import Optional

import neuron
import pandas as pd
from bluecellulab import circuit
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.config import BluepySimulationConfig
from bluecellulab.circuit.config.definition import SimulationConfig
from bluecellulab.circuit.node_id import CellId
from bluecellulab.circuit.synapse_properties import SynapseProperties, SynapseProperty, properties_from_bluepy, properties_to_bluepy

from bluecellulab.exceptions import BluecellulabError, ExtraDependencyMissingError


from bluecellulab import BLUEPY_AVAILABLE


if BLUEPY_AVAILABLE:
    import bluepy
    from bluepy.enums import Cell as BluepyCell
    from bluepy.impl.connectome_sonata import SonataConnectome

logger = logging.getLogger(__name__)


class BluepyCircuitAccess:
    """Bluepy implementation of CircuitAccess protocol."""

    def __init__(self, simulation_config: str | Path | SimulationConfig) -> None:
        """Initialize bluepy circuit object.

        BlueConfig also is a valid type.
        """
        if not BLUEPY_AVAILABLE:
            raise ExtraDependencyMissingError("bluepy")
        if isinstance(simulation_config, Path):
            simulation_config = str(simulation_config)
        if isinstance(simulation_config, str) and not Path(simulation_config).exists():
            raise FileNotFoundError(
                f"Circuit config file {simulation_config} not found.")

        if isinstance(simulation_config, BluepySimulationConfig):
            simulation_config = simulation_config.impl

        self._bluepy_sim = bluepy.Simulation(simulation_config)
        self._bluepy_circuit = self._bluepy_sim.circuit
        self.config = BluepySimulationConfig(
            self._bluepy_sim.config)

    @property
    def available_cell_properties(self) -> set:
        """Retrieve the available properties from connectome."""
        return self._bluepy_circuit.cells.available_properties

    def _get_emodel_info(self, gid: int) -> dict:
        """Return the emodel info for a gid."""
        return self._bluepy_circuit.emodels.get_mecombo_info(gid)

    def get_emodel_properties(self, cell_id: CellId) -> Optional[EmodelProperties]:
        """Get emodel_properties either from node properties or mecombo tsv."""
        gid = cell_id.id
        if self.use_mecombo_tsv:
            emodel_info = self._get_emodel_info(gid)
            emodel_properties = EmodelProperties(
                threshold_current=emodel_info["threshold_current"],
                holding_current=emodel_info["holding_current"],
                AIS_scaler=None
            )
        elif self.node_properties_available:
            cell_properties = self.get_cell_properties(
                cell_id,
                properties=[
                    "@dynamics:threshold_current",
                    "@dynamics:holding_current",
                ],
            )
            emodel_properties = EmodelProperties(
                threshold_current=float(cell_properties["@dynamics:threshold_current"]),
                holding_current=float(cell_properties["@dynamics:holding_current"]),
                AIS_scaler=None
            )
        else:  # old circuits
            return None

        if "@dynamics:AIS_scaler" in self.available_cell_properties:
            emodel_properties.AIS_scaler = float(self.get_cell_properties(
                cell_id, properties=["@dynamics:AIS_scaler"])["@dynamics:AIS_scaler"])

        if "@dynamics:soma_scaler" in self.available_cell_properties:
            emodel_properties.soma_scaler = float(self.get_cell_properties(
                cell_id, properties=["@dynamics:soma_scaler"])["@dynamics:soma_scaler"])

        return emodel_properties

    def get_template_format(self) -> Optional[str]:
        """Return the template format."""
        if self.use_mecombo_tsv or self.node_properties_available:
            return 'v6'
        else:
            return None

    def get_cell_properties(
        self, cell_id: CellId, properties: list[str] | str
    ) -> pd.Series:
        """Get a property of a cell."""
        gid = cell_id.id
        if isinstance(properties, str):
            properties = [properties]
        return self._bluepy_circuit.cells.get(gid, properties=properties)

    def get_population_ids(
        self, edge_name: str
    ) -> tuple[int, int]:
        """Retrieve the population ids of a projection."""
        projection = edge_name
        if projection in self.config.get_all_projection_names():
            if "PopulationID" in self.config.impl[f"Projection_{projection}"]:
                source_popid = int(self.config.impl[f"Projection_{projection}"]["PopulationID"])
            else:
                source_popid = 0
        else:
            source_popid = 0
        # ATM hard coded in neurodamus, commit: cd26654
        target_popid = 0
        return source_popid, target_popid

    def _get_connectomes_dict(self, projections: Optional[list[str] | str]) -> dict:
        """Get the connectomes dictionary indexed by projections or connectome
        ids."""
        if isinstance(projections, str):
            projections = [projections]

        connectomes = {'': self._bluepy_circuit.connectome}
        if projections is not None:
            proj_conns = {p: self._bluepy_circuit.projection(p) for p in projections}
            connectomes.update(proj_conns)

        return connectomes

    def extract_synapses(
        self, cell_id: CellId, projections: Optional[list[str] | str]
    ) -> pd.DataFrame:
        """Extract the synapses of a cell.

        Returns:
            synapses dataframes indexed by projection, edge and synapse ids
        """
        gid = cell_id.id
        connectomes = self._get_connectomes_dict(projections)

        all_synapses: list[pd.DataFrame] = []
        for proj_name, connectome in connectomes.items():
            connectome_properties: list[SynapseProperty | str] = list(
                SynapseProperties.common
            )

            # older circuit don't have these properties
            for test_property in [SynapseProperty.U_HILL_COEFFICIENT,
                                  SynapseProperty.CONDUCTANCE_RATIO,
                                  SynapseProperty.NRRP]:
                if test_property.to_bluepy() not in connectome.available_properties:
                    connectome_properties.remove(test_property)
                    logger.debug(f'WARNING: {test_property} not found, disabling')

            # if plasticity properties are available add them
            if all(
                x in connectome.available_properties
                for x in SynapseProperties.plasticity
            ):
                props_to_add = list(SynapseProperties.plasticity)
                connectome_properties += props_to_add

            if isinstance(connectome._impl, SonataConnectome):
                logger.debug('Using sonata style synapse file, not nrn.h5')
                connectome_properties.remove(SynapseProperty.POST_SEGMENT_OFFSET)
            else:  # afferent section_pos will be computed via post_segment_offset
                connectome_properties.remove(SynapseProperty.AFFERENT_SECTION_POS)

            connectome_properties = properties_to_bluepy(connectome_properties)
            synapses = connectome.afferent_synapses(
                gid, properties=connectome_properties
            )
            synapses.columns = properties_from_bluepy(synapses.columns)

            synapses = synapses.reset_index(drop=True)
            synapses.index = pd.MultiIndex.from_tuples(
                [(proj_name, x) for x in synapses.index],
                names=["proj_id", "synapse_id"])

            logger.info(f'Retrieving a total of {synapses.shape[0]} synapses for set {proj_name}')

            all_synapses.append(synapses)

        result = pd.concat(all_synapses)

        # add source_population_name as a column
        result["source_population_name"] = ""

        # io/synapse_reader.py:_patch_delay_fp_inaccuracies from
        # py-neurodamus
        dt = neuron.h.dt
        result[SynapseProperty.AXONAL_DELAY] = (
            result[SynapseProperty.AXONAL_DELAY] / dt + 1e-5
        ).astype('i4') * dt

        if SynapseProperty.NRRP in result:
            circuit.validate.check_nrrp_value(result)

        proj_ids: list[str] = result.index.get_level_values(0).tolist()
        pop_ids = [
            self.get_population_ids(proj_id)
            for proj_id in proj_ids
        ]
        source_popid, target_popid = zip(*pop_ids)

        result = result.assign(
            source_popid=source_popid, target_popid=target_popid
        )

        if result.empty:
            logger.warning('No synapses found')
        else:
            n_syn_sets = len(result)
            logger.info(f'Found a total of {n_syn_sets} synapse sets')

        return result

    @property
    def use_mecombo_tsv(self) -> bool:
        """Property that decides whether to use mecombo_tsv."""
        _use_mecombo_tsv = False
        if self.node_properties_available:
            _use_mecombo_tsv = False
        elif 'MEComboInfoFile' in self.config.impl.Run:
            _use_mecombo_tsv = True
        return _use_mecombo_tsv

    def target_contains_cell(self, target: str, cell_id: CellId) -> bool:
        """Check if target contains the cell or target is the cell."""
        return self._is_cell_target(target, cell_id) or self._target_has_gid(target, cell_id)

    @staticmethod
    def _is_cell_target(target: str, cell_id: CellId) -> bool:
        """Check if target is a cell."""
        return target == f"a{cell_id.id}"

    @lru_cache(maxsize=1000)
    def is_valid_group(self, group: str) -> bool:
        """Check if target is a group of cells."""
        return group in self._bluepy_circuit.cells.targets

    @lru_cache(maxsize=16)
    def get_target_cell_ids(self, target: str) -> set[CellId]:
        """Return GIDs in target as a set."""
        ids = self._bluepy_circuit.cells.ids(target)
        return {CellId("", id) for id in ids}

    @lru_cache(maxsize=1000)
    def _target_has_gid(self, target: str, cell_id: CellId) -> bool:
        """Checks if target has the gid."""
        return cell_id in self.get_target_cell_ids(target)

    @lru_cache(maxsize=100)
    def fetch_cell_info(self, cell_id: CellId) -> pd.Series:
        """Fetch bluepy cell info of a gid."""
        gid = cell_id.id
        if gid in self._bluepy_circuit.cells.ids():
            return self._bluepy_circuit.cells.get(gid)
        else:
            raise BluecellulabError(f"Gid {gid} not found in circuit")

    def _fetch_mecombo_name(self, cell_id: CellId) -> str:
        """Fetch mecombo name for a certain gid."""
        cell_info = self.fetch_cell_info(cell_id)
        if self.node_properties_available:
            me_combo = str(cell_info['model_template'])
            me_combo = me_combo.split('hoc:')[1]
        else:
            me_combo = str(cell_info['me_combo'])
        return me_combo

    def _fetch_emodel_name(self, cell_id: CellId) -> str:
        """Get the emodel path of a gid."""
        me_combo = self._fetch_mecombo_name(cell_id)
        if self.use_mecombo_tsv:
            gid = cell_id.id
            emodel_name = self._bluepy_circuit.emodels.get_mecombo_info(gid)["emodel"]
        else:
            emodel_name = me_combo

        return emodel_name

    def fetch_mini_frequencies(self, cell_id: CellId) -> tuple[float | None, float | None]:
        """Get inhibitory frequency of gid."""
        cell_info = self.fetch_cell_info(cell_id)
        # mvd uses inh_mini_frequency, sonata uses inh-mini_frequency
        inh_keys = ("inh-mini_frequency", "inh_mini_frequency")
        exc_keys = ("exc-mini_frequency", "exc_mini_frequency")

        inh_mini_frequency, exc_mini_frequency = None, None
        for inh_key in inh_keys:
            if inh_key in cell_info:
                inh_mini_frequency = cell_info[inh_key]
        for exc_key in exc_keys:
            if exc_key in cell_info:
                exc_mini_frequency = cell_info[exc_key]

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

    def get_gids_of_mtypes(self, mtypes: list[str]) -> set[CellId]:
        """Returns all the gids belonging to one of the input mtypes."""
        gids = set()
        for mtype in mtypes:
            ids = self._bluepy_circuit.cells.ids({BluepyCell.MTYPE: mtype})
            gids |= {CellId("", x) for x in ids}

        return gids

    def get_cell_ids_of_targets(self, targets: list[str]) -> set[CellId]:
        """Return all the gids belonging to one of the input targets."""
        cell_ids = set()
        for target in targets:
            cell_ids |= self.get_target_cell_ids(target)
        return cell_ids

    def morph_filepath(self, cell_id: CellId) -> str:
        return self._bluepy_circuit.morph.get_filepath(cell_id.id, "ascii")

    def emodel_path(self, cell_id: CellId) -> str:
        return os.path.join(self._emodels_dir, f"{self._fetch_emodel_name(cell_id)}.hoc")

    @property
    def _emodels_dir(self) -> str:
        return self.config.impl.Run['METypePath']
