#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import nose.tools as nt
import bglibpy

"""
def test_search_hyp_current_replay_gidlist():
    ""Tools: test search_hyp_current_replay_gidlist""
    blueconfig_location = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"
    #gids = [107462, 107461]
    gids = [107461]

    bglibpy.VERBOSE_LEVEL = 1

    results = bglibpy.search_hyp_current_replay_gidlist(blueconfig_location, gids,
        target_voltage=-80,
        min_current=-2.0,
        max_current=0.0,
        start_time=5,
        stop_time=20,
        precision=.5,
        max_nestlevel=10,
        return_fullrange=False
        )

    for gid in results:
        (step_level, voltage) = results[gid]
        print step_level, voltage
"""

def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bglibpy.calculate_SS_voltage_subprocess("examples/cell_example1/test_cell.hoc", "examples/cell_example1", 0)
    nt.assert_true(abs(SS_voltage - -73.9235504304) < 0.001)

    SS_voltage_stoch = bglibpy.calculate_SS_voltage_subprocess("examples/cell_example2/test_cell.hoc", "examples/cell_example2", 0)
    nt.assert_true(abs(SS_voltage_stoch - -73.9235504304) < 0.001)

