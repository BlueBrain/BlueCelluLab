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
import hashlib
from functools import lru_cache
import logging
from pathlib import Path
from typing import Optional

from bluepysnap.bbp import Cell as SnapCell
from bluepysnap.circuit_ids import CircuitNodeId, CircuitEdgeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import Circuit as SnapCircuit
import neuron
import pandas as pd
from bluecellulab import circuit
from bluecellulab.circuit.circuit_access.definition import EmodelProperties
from bluecellulab.circuit import CellId, SynapseProperty
from bluecellulab.circuit.config import SimulationConfig
from bluecellulab.circuit.synapse_properties import SynapseProperties
from bluecellulab.circuit.config import SimulationConfig, SonataSimulationConfig
from bluecellulab.circuit.synapse_properties import (
    properties_from_snap,
    properties_to_snap,
)

logger = logging.getLogger(__name__)


class SonataCircuitAccess:
    """Sonata implementation of CircuitAccess protocol."""

    def __init__(self, simulation_config: str | Path | SimulationConfig) -> None:
        """Initialize SonataCircuitAccess object."""
        if isinstance(simulation_config, (str, Path)) and not Path(simulation_config).exists():
            raise FileNotFoundError(f"Circuit config file {simulation_config} not found.")

        if isinstance(simulation_config, SonataSimulationConfig):
            self.config: SimulationConfig = simulation_config
        else:
            self.config = SonataSimulationConfig(simulation_config)
        circuit_config = self.config.impl.config["network"]
        self._circuit = SnapCircuit(circuit_config)

    @property
    def available_cell_properties(self) -> set:
        return self._circuit.nodes.property_names

    def get_emodel_properties(self, cell_id: CellId) -> Optional[EmodelProperties]:
        cell_properties = self._circuit.nodes[cell_id.population_name].get(cell_id.id)
        if "@dynamics:AIS_scaler" in cell_properties:
            AIS_scaler = cell_properties["@dynamics:AIS_scaler"]
        else:
            AIS_scaler = 1.0
        if "@dynamics:soma_scaler" in cell_properties:
            soma_scaler = cell_properties["@dynamics:soma_scaler"]
        else:
            soma_scaler = 1.0

        return EmodelProperties(
            cell_properties["@dynamics:threshold_current"],
            cell_properties["@dynamics:holding_current"],
            AIS_scaler,
            soma_scaler,
        )

    def get_template_format(self) -> Optional[str]:
        return 'v6'

    def get_cell_properties(
        self, cell_id: CellId, properties: list[str] | str
    ) -> pd.Series:
        if isinstance(properties, str):
            properties = [properties]
        return self._circuit.nodes[cell_id.population_name].get(
            cell_id.id, properties=properties
        )

    @staticmethod
    def _compute_pop_ids(source: str, target: str) -> tuple[int, int]:
        """Compute the population ids from the population names."""
        def make_id(node_pop: str) -> int:
            pop_hash = hashlib.md5(node_pop.encode()).digest()
            return ((pop_hash[1] & 0x0f) << 8) + pop_hash[0]  # id: 12bit hash

        source_popid = make_id(source)
        target_popid = make_id(target)
        return source_popid, target_popid

    def get_population_ids(
        self, source_population_name: str, target_population_name: str
    ) -> tuple[int, int]:
        source_popid, target_popid = self._compute_pop_ids(
            source_population_name, target_population_name)
        return source_popid, target_popid

    def extract_synapses(
        self, cell_id: CellId, projections: Optional[list[str] | str]
    ) -> pd.DataFrame:
        """Extract the synapses.

        If projections is None, all the synapses are extracted.
        """
        snap_node_id = CircuitNodeId(cell_id.population_name, cell_id.id)
        edges = self._circuit.edges
        # select edges that are in the projections, if there are projections
        if projections is None or len(projections) == 0:
            edge_population_names = [x for x in edges]
        elif isinstance(projections, str):
            edge_population_names = [x for x in edges if edges[x].source.name == projections]
        else:
            edge_population_names = [x for x in edges if edges[x].source.name in projections]

        all_synapses_dfs: list[pd.DataFrame] = []
        for edge_population_name in edge_population_names:
            edge_population = edges[edge_population_name]
            afferent_edges: CircuitEdgeIds = edge_population.afferent_edges(snap_node_id)
            if len(afferent_edges) != 0:
                # first copy the common properties to modify them
                edge_properties: list[SynapseProperty | str] = list(
                    SynapseProperties.common
                )

                # remove optional properties if they are not present
                for optional_property in [SynapseProperty.U_HILL_COEFFICIENT,
                                          SynapseProperty.CONDUCTANCE_RATIO]:
                    if optional_property.to_snap() not in edge_population.property_names:
                        edge_properties.remove(optional_property)

                # if all plasticity props are present, add them
                if all(
                    x in edge_population.property_names
                    for x in SynapseProperties.plasticity
                ):
                    edge_properties += list(SynapseProperties.plasticity)

                snap_properties = properties_to_snap(edge_properties)
                synapses: pd.DataFrame = edge_population.get(afferent_edges, snap_properties)
                column_names = list(synapses.columns)
                synapses.columns = pd.Index(properties_from_snap(column_names))

                # make multiindex
                synapses = synapses.reset_index(drop=True)
                synapses.index = pd.MultiIndex.from_tuples(
                    zip([edge_population_name] * len(synapses), synapses.index),
                    names=["edge_name", "synapse_id"],
                )

                # add source_population_name as a column
                synapses["source_population_name"] = edges[edge_population_name].source.name

                # py-neurodamus
                dt = neuron.h.dt
                synapses[SynapseProperty.AXONAL_DELAY] = (
                    synapses[SynapseProperty.AXONAL_DELAY] / dt + 1e-5
                ).astype('i4') * dt

                if SynapseProperty.NRRP in synapses:
                    circuit.validate.check_nrrp_value(synapses)

                source_popid, target_popid = self.get_population_ids(
                    edge_population.source.name, edge_population.target.name)
                synapses = synapses.assign(
                    source_popid=source_popid, target_popid=target_popid
                )

                all_synapses_dfs.append(synapses)

        if len(all_synapses_dfs) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(all_synapses_dfs)  # outer join that creates NaNs

    def target_contains_cell(self, target: str, cell_id: CellId) -> bool:
        return cell_id in self.get_target_cell_ids(target)

    @lru_cache(maxsize=1000)
    def is_valid_group(self, group: str) -> bool:
        return group in self._circuit.node_sets

    @lru_cache(maxsize=16)
    def get_target_cell_ids(self, target: str) -> set[CellId]:
        ids = self._circuit.nodes.ids(target)
        return {CellId(x.population, x.id) for x in ids}

    @lru_cache(maxsize=100)
    def fetch_cell_info(self, cell_id: CellId) -> pd.Series:
        return self._circuit.nodes[cell_id.population_name].get(cell_id.id)

    def fetch_mini_frequencies(self, cell_id: CellId) -> tuple[float | None, float | None]:
        cell_info = self.fetch_cell_info(cell_id)
        exc_mini_frequency = cell_info['exc-mini_frequency'] \
            if 'exc-mini_frequency' in cell_info else None
        inh_mini_frequency = cell_info['inh-mini_frequency'] \
            if 'inh-mini_frequency' in cell_info else None
        return exc_mini_frequency, inh_mini_frequency

    @property
    def node_properties_available(self) -> bool:
        return True

    def get_gids_of_mtypes(self, mtypes: list[str]) -> set[CellId]:
        all_cell_ids = set()
        all_population_names: list[str] = list(self._circuit.nodes)
        for population_name in all_population_names:
            try:
                cell_ids = self._circuit.nodes[population_name].ids(
                    {SnapCell.MTYPE: mtypes})
            except BluepySnapError:
                continue
            all_cell_ids |= {CellId(population_name, id) for id in cell_ids}
        return all_cell_ids

    def get_cell_ids_of_targets(self, targets: list[str]) -> set[CellId]:
        cell_ids = set()
        for target in targets:
            cell_ids |= self.get_target_cell_ids(target)
        return cell_ids

    def morph_filepath(self, cell_id: CellId) -> str:
        """Returns the .asc morphology path from 'alternate_morphologies'."""
        node_population = self._circuit.nodes[cell_id.population_name]
        try:  # if asc defined in alternate morphology
            return str(node_population.morph.get_filepath(cell_id.id, extension="asc"))
        except BluepySnapError as e:
            logger.debug(f"No asc morphology found for {cell_id}, trying swc.")
            return str(node_population.morph.get_filepath(cell_id.id))

    def emodel_path(self, cell_id: CellId) -> str:
        node_population = self._circuit.nodes[cell_id.population_name]
        return str(node_population.models.get_filepath(cell_id.id))
