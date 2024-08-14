# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the neuron interpreter."""

import os

import neuron
from pytest import raises
import pytest

import bluecellulab
from bluecellulab.exceptions import NeuronEvalError
from bluecellulab.importer import _load_mod_files
from bluecellulab.neuron_interpreter import eval_neuron

script_dir = os.path.dirname(__file__)


def test_eval_neuron():
    """Unit test for the eval_neuron function."""
    _load_mod_files()
    eval_neuron("neuron.h.nil", neuron=neuron)
    with raises(NeuronEvalError):
        eval_neuron("1+1")
    with raises(NeuronEvalError):
        eval_neuron("neuron.h.nil; 2-1", neuron=neuron)
    with raises(NeuronEvalError):
        eval_neuron("a=1")


@pytest.mark.v5
def test_eval_neuron_with_cell():
    """Test the eval neuron function using a cell."""
    cell = bluecellulab.Cell(
        f"{script_dir}/examples/cell_example1/test_cell.hoc",
        f"{script_dir}/examples/cell_example1",
    )

    eval_neuron("self.axonal[1](0.5)._ref_v", self=cell)
    AXON_LOC = "self.axonal[1](0.5)._ref_v"
    cell.add_recordings(["neuron.h._ref_t", AXON_LOC], dt=cell.record_dt)
