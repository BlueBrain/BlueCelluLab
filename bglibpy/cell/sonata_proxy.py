"""Interfacing the cell object with SONATA properties."""

from functools import partialmethod
from typing import Any
from bglibpy.circuit import CircuitAccess
from bglibpy.circuit.node_id import CellId
from bglibpy.exceptions import MissingSonataPropertyError


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
