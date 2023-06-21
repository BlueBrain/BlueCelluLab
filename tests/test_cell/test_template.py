"""Unit tests for template.py module."""

from pathlib import Path

import pytest

from bluecellulab.cell.template import NeuronTemplate


parent_dir = Path(__file__).resolve().parent.parent

hoc_path = (
    parent_dir
    / "examples"
    / "hippocampus_opt_cell_template"
    / "electrophysiology"
    / "cell.hoc"
)
morph_path = (
    parent_dir / "examples" / "hippocampus_opt_cell_template" / "morphology" / "cell.asc"
)


def test_get_cell_with_bluepyopt_template():
    """Unit test for the get_cell method with bluepyopt_template."""
    template = NeuronTemplate(hoc_path, morph_path)
    cell = template.get_cell("bluepyopt", None, None)
    assert cell.hname() == "bACnoljp_bluecellulab[0]"


def test_neuron_template_init():
    """Unit test for the NeuronTemplate's constructor."""
    missing_file = "missing_file"

    with pytest.raises(FileNotFoundError):
        NeuronTemplate(missing_file, morph_path)
    with pytest.raises(FileNotFoundError):
        NeuronTemplate(hoc_path, missing_file)

    NeuronTemplate(hoc_path, morph_path)
