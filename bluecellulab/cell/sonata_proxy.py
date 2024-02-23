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
"""Interfacing the cell object with SONATA properties."""

from functools import partialmethod
from typing import Any
from bluecellulab.circuit import CircuitAccess
from bluecellulab.circuit.node_id import CellId
from bluecellulab.exceptions import MissingSonataPropertyError


class SonataProxy:
    """Sonata proxy to be used by Cell objects."""

    def __init__(self, cell_id: CellId, circuit_access: CircuitAccess) -> None:
        self.cell_id = cell_id
        self.circuit_access = circuit_access

    def get_property(self, property_name: str) -> Any:
        """Get a property of a cell."""
        if property_name in self.circuit_access.available_cell_properties:
            return self.circuit_access.get_cell_properties(self.cell_id, property_name)
        raise MissingSonataPropertyError(f"{property_name} property is not available.")

    get_input_resistance = partialmethod(get_property, "@dynamics:input_resistance")
