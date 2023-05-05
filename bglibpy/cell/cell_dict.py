"""Dictionary of cells that support the old way of retrieval by int ids.
   as well as the new way of retrieval by CellId."""
from __future__ import annotations

import bglibpy
from bglibpy.circuit.node_id import CellId, create_cell_id


class CellDict(dict):
    """Dictionary of Cell objects, backwards compatible with old gid representation."""

    def __setitem__(self, key: int | tuple[str, int], value: bglibpy.Cell) -> None:
        super().__setitem__(create_cell_id(key), value)

    def __getitem__(self, key: int | tuple[str, int]) -> bglibpy.Cell:
        """Can retrieve for single population circuits, supports cells[gid]."""
        cell_id: CellId = create_cell_id(key)
        return super().__getitem__(cell_id)
