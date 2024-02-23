# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
