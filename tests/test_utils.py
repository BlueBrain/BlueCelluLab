import json

import numpy as np
from bluecellulab.utils import CaptureOutput, IsolatedProcess, NumpyEncoder, run_once


# Decorated function for testing
@run_once
def increment_counter(counter):
    counter[0] += 1
    return "Executed"


def test_run_once_execution():
    """Test that the decorated function runs only once."""
    counter = [0]  # Using a list for mutability

    assert increment_counter(counter) == "Executed"
    increment_counter(counter)
    assert counter[0] == 1

    # Called 3 times but increased once
    increment_counter(counter)
    increment_counter(counter)
    increment_counter(counter)

    assert counter[0] == 1

    assert increment_counter(counter) is None


def test_capture_output():
    with CaptureOutput() as output:
        print("Hello, World!")
        print("This is a test.")

    assert len(output) == 2
    assert "Hello, World!" in output
    assert "This is a test." in output


def test_no_output():
    with CaptureOutput() as output:
        pass  # No output

    assert len(output) == 0


def test_numpy_encoder():
    """Utils: Test NumpyEncoder"""
    assert json.dumps(np.int32(1), cls=NumpyEncoder) == "1"
    assert json.dumps(np.float32(1.2), cls=NumpyEncoder)[0:3] == "1.2"
    assert json.dumps(np.array([1, 2, 3]), cls=NumpyEncoder) == "[1, 2, 3]"
    assert json.dumps(np.array([1.2, 2.3, 3.4]), cls=NumpyEncoder) == "[1.2, 2.3, 3.4]"
    assert (
        json.dumps(np.array([True, False, True]), cls=NumpyEncoder)
        == "[true, false, true]"
    )


def test_isolated_process():
    """Test to ensure isolated process keeps its properties."""
    runner = IsolatedProcess()
    assert runner._processes == 1
    assert runner._maxtasksperchild == 1
