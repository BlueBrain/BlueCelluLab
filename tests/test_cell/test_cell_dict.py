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
"""Unit tests for the cell_dict module."""
from pathlib import Path

import pytest

from bluecellulab.cell import Cell, CellDict
from bluecellulab.circuit.node_id import CellId


parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v5
def test_cell_dict():
    """Unit test for the CellDict class."""
    cell = Cell(
        "%s/examples/cell_example1/test_cell_longname1.hoc" % str(parent_dir),
        "%s/examples/cell_example1" % str(parent_dir))

    cell_dict = CellDict()
    cell_dict[1] = cell
    assert cell_dict[("", 1)] == cell
    assert cell_dict[1] == cell

    cell_dict[("population1", 2)] = cell
    cell_id = CellId("population1", 2)
    assert cell_dict[cell_id] == cell
    assert cell_dict[("population1", 2)] == cell
