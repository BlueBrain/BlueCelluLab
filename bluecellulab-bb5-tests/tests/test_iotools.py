"""Unit tests for the iotools module of circuit."""

import pathlib

import pytest

from bluecellulab.circuit.iotools import parse_outdat
from bluecellulab.circuit.node_id import CellId


script_dir = pathlib.Path(__file__).resolve().parent


def test_parse_outdat():
    """SSim: Testing parsing of out.dat"""
    cell_id = CellId("", 2)
    with pytest.raises(IOError):
        parse_outdat(
            script_dir / "examples/sim_twocell_empty/output_doesntexist/out.dat"
        )

    outdat = parse_outdat(
        script_dir / "examples/sim_twocell_minis_replay/output/out.dat"
    )
    assert 45 in outdat[cell_id]

    outdat_with_negatives = parse_outdat(
        script_dir
        / "examples/sim_twocell_minis_replay/output/out-contains-negatives.dat"
    )
    assert all(outdat_with_negatives[cell_id] >= 0)
