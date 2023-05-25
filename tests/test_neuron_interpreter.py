"""Unit tests for the neuron interpreter."""

import os

from pytest import raises
import pytest

import bluecellulab
from bluecellulab.exceptions import NeuronEvalError
from bluecellulab.neuron_interpreter import eval_neuron

script_dir = os.path.dirname(__file__)


def test_eval_neuron():
    """Unit test for the eval_neuron function."""
    eval_neuron("bluecellulab.neuron.h.nil", bluecellulab=bluecellulab)
    with raises(NeuronEvalError):
        eval_neuron("1+1")
    with raises(NeuronEvalError):
        eval_neuron("bluecellulab.neuron.h.nil; 2-1", bluecellulab=bluecellulab)
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
