#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import json
import os

import numpy as np
from pytest import approx

import bglibpy
from bglibpy.tools import NumpyEncoder, Singleton
from tests.helpers.circuit import blueconfig_append_path

script_dir = os.path.dirname(__file__)


def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bglibpy.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir,
        0)
    assert abs(SS_voltage - -73.9235504304) < 0.001

    SS_voltage_stoch = bglibpy.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example2/test_cell.hoc" % script_dir,
        "%s/examples/cell_example2" % script_dir,
        0)
    assert abs(SS_voltage_stoch - -73.9235504304) < 0.001


class TestTools:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_empty"
        )
        blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

        self.config = blueconfig_append_path(
            blueconfig_path, conf_pre_path
        )

    def test_holding_current(self):
        """Tools: Test holding_current"""

        gid = 1
        expected_voltage = -80
        for expected_current, ttx in [
            (-0.08019584734739738, False),
                (-0.08019289690395226, True)]:
            holding_current, holding_voltage = bglibpy.tools.holding_current(
                expected_voltage, gid, self.config, enable_ttx=ttx)
            holding_current == approx(expected_current, abs=1e-6)
            holding_voltage == approx(expected_voltage, abs=1e-6)


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
