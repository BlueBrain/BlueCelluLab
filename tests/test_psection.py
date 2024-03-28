"""Unit tests for the psection module."""

import pytest
from pathlib import Path
from bluecellulab import Cell
from bluecellulab.circuit.circuit_access.definition import EmodelProperties
from bluecellulab.psection import init_psections
from bluecellulab.cell.template import public_hoc_cell


parent_dir = Path(__file__).resolve().parent


class TestPSection:
    """Test class for testing Cell object functionalities with v6 template."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup."""
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0
        )
        self.cell = Cell(
            f"{parent_dir}/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
            f"{parent_dir}/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
            template_format="v6",
            emodel_properties=emodel_properties
        )
        self.psections, self.secname_to_psection = init_psections(public_hoc_cell(self.cell.cell))

    def test_is_leaf(self):
        """Test the is_leaf property for leaf and non-leaf sections."""
        leaf_section = self.psections[245]
        non_leaf_section = self.psections[1]
        assert leaf_section.is_leaf
        assert not non_leaf_section.is_leaf

    def test_hparent_and_hchildren(self):
        """Test hparent and hchildren properties for a section with a parent and children."""
        root = self.psections[0]
        assert root.hparent is None
        a_section = self.psections[145]
        assert a_section.hparent is not None
        assert len(a_section.hchildren) > 0
        leaf_section = self.psections[245]
        assert len(leaf_section.hchildren) == 0

    def test_add_pchild(self):
        """Test adding a child section."""
        parent_section = self.psections[245]  # leaf
        assert len(parent_section.pchildren) == 0
        child_section = self.psections[244]  # leaf
        parent_section.add_pchild(child_section)
        assert child_section in parent_section.pchildren
        assert child_section.pparent == parent_section

    def test_all_descendants(self):
        """Test retrieving all descendants of a section."""
        root_section = self.psections[0]
        all_descendants = root_section.all_descendants()
        # Excluding the root and myelin sections: -1 + -1 = -2
        assert len(all_descendants) == len(self.psections) - 2

    def test_tree_width(self):
        """Test calculation of tree width."""
        tree_width = self.psections[0].tree_width()
        assert tree_width > 0

    def test_tree_height(self):
        """Test calculation of tree height."""
        tree_height = self.psections[0].tree_height()
        assert tree_height > 0

    def test_init_psections(self):
        """Test if psections and secname_to_psection are properly initialized."""
        assert len(self.psections) > 0
        assert len(self.secname_to_psection) > 0
        assert len(self.psections) == len(self.secname_to_psection)
