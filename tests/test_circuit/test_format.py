"""Unit tests for circuit::format.py."""
from pathlib import Path
from bluecellulab.circuit.format import CircuitFormat, determine_circuit_format, is_valid_json_file
from bluecellulab.circuit.config.simulation_config import SonataSimulationConfig

script_dir = Path(__file__).resolve().parent.parent

sonata_conf = (
    script_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config.json"
)
blueconfig = (
    script_dir
    / "examples"
    / "legacy_circuitconfig"
    / "CircuitConfig"
)


def test_determine_circuit_format():
    """Test the determine_circuit_format function."""
    assert determine_circuit_format(SonataSimulationConfig(sonata_conf)) == CircuitFormat.SONATA
    assert determine_circuit_format(sonata_conf) == CircuitFormat.SONATA
    assert determine_circuit_format(blueconfig) == CircuitFormat.BLUECONFIG


def test_is_valid_json_file():
    """Test the is_valid_json_file function."""
    assert not is_valid_json_file(blueconfig)
    assert not is_valid_json_file({1: 5, 2: 6})
    assert is_valid_json_file(sonata_conf)
