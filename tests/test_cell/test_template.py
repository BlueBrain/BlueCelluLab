"""Unit tests for template.py module."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from bluecellulab.cell.template import NeuronTemplate, public_hoc_cell
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.exceptions import BluecellulabError


parent_dir = Path(__file__).resolve().parent.parent

hipp_hoc_path = (
    parent_dir
    / "examples"
    / "hippocampus_opt_cell_template"
    / "electrophysiology"
    / "cell.hoc"
)
hipp_morph_path = (
    parent_dir / "examples" / "hippocampus_opt_cell_template" / "morphology" / "cell.asc"
)

v6_hoc_path = (
    parent_dir / "examples" / "circuit_sonata_quick_scx" / "components" / "hoc" / "cADpyr_L2TPC.hoc"
)

v6_morph_path = (
    parent_dir / "examples" / "circuit_sonata_quick_scx" / "components" / "morphologies" / "asc" / "rr110330_C3_idA.asc"
)


def test_get_cell_with_bluepyopt_template():
    """Unit test for the get_cell method with bluepyopt_template."""
    template = NeuronTemplate(hipp_hoc_path, hipp_morph_path)
    cell = template.get_cell("bluepyopt", None, None)
    assert cell.hname() == f"bACnoljp_bluecellulab_{(hex(id(template)))}[0]"


def test_neuron_template_init():
    """Unit test for the NeuronTemplate's constructor."""
    missing_file = "missing_file"

    with pytest.raises(FileNotFoundError):
        NeuronTemplate(missing_file, hipp_morph_path)
    with pytest.raises(FileNotFoundError):
        NeuronTemplate(hipp_hoc_path, missing_file)

    NeuronTemplate(hipp_hoc_path, hipp_morph_path)


def test_public_hoc_cell_bluepyopt_template():
    """Unit test for public_hoc_cell."""
    template = NeuronTemplate(hipp_hoc_path, hipp_morph_path)
    cell = template.get_cell("bluepyopt", None, None)
    hoc_public = public_hoc_cell(cell)
    assert hoc_public.gid == 0.0


def test_public_hoc_cell_v6_template():
    """Unit test for public_hoc_cell."""
    template = NeuronTemplate(v6_hoc_path, v6_morph_path)
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    cell = template.get_cell("v6", 5, emodel_properties)
    hoc_public = public_hoc_cell(cell)
    assert hoc_public.gid == 5.0


def test_public_hoc_cell_failure():
    """Unit test for public_hoc_cell when neither getCell nor CellRef is provided."""
    cell_without_getCell_or_CellRef = Mock(spec=[])  # spec=[] ensures no attributes exist
    with pytest.raises(BluecellulabError) as excinfo:
        public_hoc_cell(cell_without_getCell_or_CellRef)
    assert "Public cell properties cannot be accessed" in str(excinfo.value)
