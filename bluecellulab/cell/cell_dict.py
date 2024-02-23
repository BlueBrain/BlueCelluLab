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
"""Dictionary of cells that support the old way of retrieval by int ids.

as well as the new way of retrieval by CellId.
"""

from __future__ import annotations

import bluecellulab
from bluecellulab.circuit.node_id import CellId, create_cell_id


class CellDict(dict):
    """Dictionary of Cell objects, backwards compatible with old gid
    representation."""

    def __setitem__(self, key: int | tuple[str, int], value: bluecellulab.Cell) -> None:
        super().__setitem__(create_cell_id(key), value)

    def __getitem__(self, key: int | tuple[str, int]) -> bluecellulab.Cell:
        """Can retrieve for single population circuits, supports cells[gid]."""
        cell_id: CellId = create_cell_id(key)
        return super().__getitem__(cell_id)
