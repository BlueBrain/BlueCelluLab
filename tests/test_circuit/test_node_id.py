# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for node_id.py"""
from bluecellulab.circuit.node_id import create_cell_id, create_cell_ids, CellId


def test_create_cell_id():
    """Testing creation of CellId."""
    assert create_cell_id(1) == CellId("", 1)
    assert create_cell_id(("", 2)) == CellId("", 2)
    assert create_cell_id(("pop1", 2)) == CellId("pop1", 2)


def test_create_cell_ids():
    """Testing creation of CellIds."""
    assert create_cell_ids([1, 2]) == [CellId("", 1), CellId("", 2)]
    assert create_cell_ids([("", 2), ("pop1", 2)]) == [
        CellId("", 2),
        CellId("pop1", 2),
    ]
