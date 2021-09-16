#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import os
from pytest import approx
import bglibpy

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


def test_blueconfig_append_path():
    """Tools: Test blueconfig_append_path."""
    conf_pre_path = os.path.join(
        script_dir, "examples", "sim_twocell_empty"
    )
    blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

    fields = ["MorphologyPath", "METypePath", "CircuitPath",
              "nrnPath", "CurrentDir", "OutputRoot", "TargetFile"]

    modified_config = bglibpy.tools.blueconfig_append_path(
        blueconfig_path, conf_pre_path, fields=fields
    )

    for field in fields:
        field_val = modified_config.Run.__getattr__(field)
        assert os.path.isabs(field_val)


class TestTools(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_empty"
        )
        blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

        self.config = bglibpy.tools.blueconfig_append_path(
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
