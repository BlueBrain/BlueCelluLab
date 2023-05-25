"""Unit tests for serialized_sections module."""
from pathlib import Path

import pytest

from bluecellulab import Cell
from bluecellulab.cell.serialized_sections import SerializedSections


script_dir = Path(__file__).parent.parent


@pytest.mark.v5
def test_serialized_sections():
    """Test the SerializedSections class."""
    cell = Cell(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir)

    serialized_sections = SerializedSections(cell.cell.getCell())
    assert len(serialized_sections.isec2sec) == 135
    assert serialized_sections.isec2sec[0].nchild() == 7
    assert serialized_sections.isec2sec[0].has_parent() is False
    assert serialized_sections.isec2sec[0].has_trueparent() is False
    for sec in serialized_sections.isec2sec.values():
        assert sec.exists()

    # check edge cases
    modified_cell = cell.cell.getCell()

    # get the first section of modified_cell
    section_list = modified_cell.all
    first_section = next(iter(section_list))
    first_section(0.0001).v = -2
    with pytest.warns(UserWarning):
        SerializedSections(modified_cell)

    modified_cell.nSecAll = -1
    with pytest.raises(ValueError):
        SerializedSections(modified_cell)
