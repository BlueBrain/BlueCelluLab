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
"""Layer to abstract the circuit access functionality from the rest of
bluecellulab."""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional, Protocol


import pandas as pd
from pydantic.dataclasses import dataclass

from bluecellulab.circuit import CellId
from bluecellulab.circuit.config import SimulationConfig


@dataclass(config=dict(extra="forbid"))
class EmodelProperties:
    threshold_current: float
    holding_current: float
    AIS_scaler: Optional[float] = 1.0
    soma_scaler: Optional[float] = 1.0


def get_synapse_connection_parameters(
        circuit_access: CircuitAccess, pre_cell: CellId, post_cell: CellId) -> dict:
    """Apply connection blocks in order for pre_gid, post_gid to determine a
    final connection override for this pair (pre_gid, post_gid)."""
    parameters: defaultdict[str, Any] = defaultdict(list)
    parameters['add_synapse'] = True

    for entry in circuit_access.config.connection_entries():
        src_matches = circuit_access.target_contains_cell(entry.source, pre_cell)
        dest_matches = circuit_access.target_contains_cell(entry.target, post_cell)

        if src_matches and dest_matches:
            # whatever specified in this block, is applied to gid
            apply_parameters = True

            if entry.delay is not None:
                parameters['DelayWeights'].append((entry.delay, entry.weight))
                apply_parameters = False

            if apply_parameters:
                if entry.weight is not None:
                    parameters['Weight'] = entry.weight
                if entry.spont_minis is not None:
                    parameters['SpontMinis'] = entry.spont_minis
                if entry.synapse_configure is not None:
                    # collect list of applicable configure blocks to be
                    # applied with a "hoc exec" statement
                    parameters['SynapseConfigure'].append(entry.synapse_configure)
                if entry.mod_override is not None:
                    parameters['ModOverride'] = entry.mod_override
    return parameters


class CircuitAccess(Protocol):
    """Protocol that defines the circuit access layer."""

    config: SimulationConfig

    @property
    def available_cell_properties(self) -> set:
        raise NotImplementedError

    def get_emodel_properties(self, cell_id: CellId) -> Optional[EmodelProperties]:
        raise NotImplementedError

    def get_template_format(self) -> Optional[str]:
        raise NotImplementedError

    def get_cell_properties(
        self, cell_id: CellId, properties: list[str] | str
    ) -> pd.Series:
        raise NotImplementedError

    def extract_synapses(
        self, cell_id: CellId, projections: Optional[list[str] | str]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def target_contains_cell(self, target: str, cell_id: CellId) -> bool:
        raise NotImplementedError

    def is_valid_group(self, group: str) -> bool:
        raise NotImplementedError

    def get_target_cell_ids(self, target: str) -> set[CellId]:
        raise NotImplementedError

    def fetch_cell_info(self, cell_id: CellId) -> pd.Series:
        raise NotImplementedError

    def fetch_mini_frequencies(self, cell_id: CellId) -> tuple[float | None, float | None]:
        raise NotImplementedError

    @property
    def node_properties_available(self) -> bool:
        raise NotImplementedError

    def get_gids_of_mtypes(self, mtypes: list[str]) -> set[CellId]:
        raise NotImplementedError

    def get_cell_ids_of_targets(self, targets: list[str]) -> set[CellId]:
        raise NotImplementedError

    def morph_filepath(self, cell_id: CellId) -> str:
        raise NotImplementedError

    def emodel_path(self, cell_id: CellId) -> str:
        raise NotImplementedError
