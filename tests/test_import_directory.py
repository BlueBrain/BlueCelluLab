"""This test file is to ensure mechanisms are loaded where Cell is created."""
import os
from pathlib import Path


def test_import_vs_first_call_neuron_mechanisms_dir():
    """Test if the neuron import directory is set at import or at the first call."""
    os.chdir("tests")  # there are no mechanisms here

    # intentionally imported in the directory without compiled mechanisms
    from bluecellulab import Cell  # noqa: E402

    examples_dir = Path(__file__).resolve().parent / "examples"

    os.chdir("..")  # mechanisms at .. should be loaded
    hoc_path = examples_dir / "bpo_cell" / "0_cADpyr_L5TPC_a6e707a_1_sNone.hoc"
    morph_path = examples_dir / "bpo_cell" / "C060114A5.asc"
    _ = Cell(hoc_path, morph_path, template_format="bluepyopt")
