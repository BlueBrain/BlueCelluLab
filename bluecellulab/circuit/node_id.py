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
"""Identifier for the circuit nodes.

Account for multi-population and the single.
"""

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
