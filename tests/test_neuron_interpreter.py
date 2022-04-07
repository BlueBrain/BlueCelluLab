"""Unit tests for the neuron interpreter."""

import os

from pytest import raises

import bglibpy
from bglibpy.exceptions import NeuronEvalError
from bglibpy.neuron_interpreter import eval_neuron

script_dir = os.path.dirname(__file__)

def test_eval_neuron():
    """Unit test for the eval_neuron function."""
    eval_neuron("bglibpy.neuron.h.nil", bglibpy=bglibpy)
    with raises(NeuronEvalError):
        eval_neuron("1+1")
    with raises(NeuronEvalError):
        eval_neuron("bglibpy.neuron.h.nil; 2-1", bglibpy=bglibpy)
    with raises(NeuronEvalError):
        eval_neuron("a=1")

def test_eval_neuron_with_cell():
    """Test the eval neuron function using a cell."""
    cell = bglibpy.Cell(
        f"{script_dir}/examples/cell_example1/test_cell.hoc",
        f"{script_dir}/examples/cell_example1",
    )

    eval_neuron("self.axonal[1](0.5)._ref_v", self=cell)
