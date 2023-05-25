"""Representing the circuit format."""
from __future__ import annotations
from enum import Enum
import json
from pathlib import Path

from bluecellulab.circuit.config import BluepySimulationConfig, SimulationConfig, SonataSimulationConfig


class CircuitFormat(Enum):
    SONATA = "sonata"
    BLUECONFIG = "blueconfig"


def determine_circuit_format(circuit: str | Path | SimulationConfig) -> CircuitFormat:
    """Returns the circuit format for the input circuit path."""
    if isinstance(circuit, SonataSimulationConfig):
        return CircuitFormat.SONATA
    elif isinstance(circuit, BluepySimulationConfig):
        return CircuitFormat.BLUECONFIG
    else:  # not possibly a SimulationConfig
        if is_valid_json_file(circuit):  # type: ignore
            return CircuitFormat.SONATA
        else:
            return CircuitFormat.BLUECONFIG


def is_valid_json_file(fpath: str | Path) -> bool:
    """Check if the input file is a valid json."""
    try:
        json.load(open(fpath))
    except (TypeError, json.JSONDecodeError):
        return False
    return True
