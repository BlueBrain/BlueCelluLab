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
