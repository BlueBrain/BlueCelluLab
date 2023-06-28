#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import json
import os

import numpy as np
import pytest

import bluecellulab
from bluecellulab.tools import NumpyEncoder, Singleton

script_dir = os.path.dirname(__file__)


@pytest.mark.v5
def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bluecellulab.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir,
        0)
    assert abs(SS_voltage - -73.9235504304) < 0.001

    SS_voltage_stoch = bluecellulab.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example2/test_cell.hoc" % script_dir,
        "%s/examples/cell_example2" % script_dir,
        0)
    assert abs(SS_voltage_stoch - -73.9235504304) < 0.001


def test_singleton():
    """Make sure only 1 object gets created in a singleton."""

    class TestClass(metaclass=Singleton):
        """Class to test Singleton object creation."""

        n_init_calls = 0

        def __init__(self):
            print("I'm called but not re-instantiated")
            TestClass.n_init_calls += 1

    test_obj1 = TestClass()
    test_obj2 = TestClass()
    test_objs = [TestClass() for _ in range(10)]

    assert test_obj1 is test_obj2
    assert id(test_obj1) == id(test_obj2)

    assert len(set(test_objs)) == 1

    assert TestClass.n_init_calls == 12


def test_numpy_encoder():
    """Tools: Test NumpyEncoder"""
    assert json.dumps(np.int32(1), cls=NumpyEncoder) == "1"
    assert json.dumps(np.float32(1.2), cls=NumpyEncoder)[0:3] == "1.2"
    assert json.dumps(np.array([1, 2, 3]), cls=NumpyEncoder) == "[1, 2, 3]"
    assert json.dumps(np.array([1.2, 2.3, 3.4]), cls=NumpyEncoder) == "[1.2, 2.3, 3.4]"
    assert (
        json.dumps(np.array([True, False, True]), cls=NumpyEncoder)
        == "[true, false, true]"
    )
