#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import nose.tools as nt
import bglibpy

def test_calculate_SS_voltage():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bglibpy.calculate_SS_voltage("examples/cell_example1/test_cell.hoc", "examples/cell_example1", 0)
    nt.assert_true(abs(SS_voltage - -73.9235504304) < 0.001)

    SS_voltage_stoch = bglibpy.calculate_SS_voltage("examples/cell_example2/test_cell.hoc", "examples/cell_example2", 0)
    nt.assert_true(abs(SS_voltage_stoch - -73.9235504304) < 0.001)
