"""Unit tests for the iotools module of circuit."""

import pathlib

import pytest

from bglibpy.circuit import parse_outdat


script_dir = pathlib.Path(__file__).resolve().parent.parent


def test_parse_outdat():
    """SSim: Testing parsing of out.dat"""
    with pytest.raises(IOError):
        parse_outdat(script_dir / "examples/sim_twocell_empty/output_doesntexist/out.dat")

    outdat = parse_outdat(
        script_dir / "examples/sim_twocell_minis_replay/output/out.dat")
    assert 45 in outdat[2]

    outdat_with_negatives = parse_outdat(
        script_dir / "examples/sim_twocell_minis_replay/output/out-contains-negatives.dat")
    assert all(outdat_with_negatives[2] >= 0)
