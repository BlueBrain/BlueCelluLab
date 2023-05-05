"""Identifier for the circuit nodes. Account for multi-population and the single."""
from __future__ import annotations
from typing import NamedTuple


CellId = NamedTuple("CellId", [("population_name", str), ("id", int)])


def create_cell_id(cell_id: int | tuple[str, int] | CellId) -> CellId:
    """Make a CellId from a tuple or an int."""
    if isinstance(cell_id, tuple):
        return CellId(*cell_id)
    else:
        return CellId("", cell_id)


def create_cell_ids(cell_ids: list[int] | list[tuple[str, int] | CellId]) -> list[CellId]:
    """Make a list of CellId from a list of tuple or int."""
    return [create_cell_id(cell_id) for cell_id in cell_ids]
