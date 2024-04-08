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
