#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import os
import bglibpy

script_dir = os.path.dirname(__file__)

import nose.tools as nt
from nose.plugins.attrib import attr

'''
@attr('bgscratch')
def test_search_hyp_current_replay_gidlist():
    """Tools: Test search_hyp_current_replay_gidlist"""
    blueconfig_location = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"
    #gids = [107462, 107461]
    gid = 107461
    precision = .5
    target_voltage = -77
    start_time = 1
    stop_time = 5

    results = bglibpy.search_hyp_current_replay_gidlist(
        blueconfig_location,
        [gid],
        target_voltage=target_voltage,
        min_current=-2.0,
        max_current=0.0,
        start_time=start_time,
        stop_time=stop_time,
        precision=precision,
        max_nestlevel=5,
        return_fullrange=False)

    nt.assert_true(gid in results)
    step_level, (time, voltage) = results[gid]
    nt.assert_equal(step_level, -1.5)
    import numpy
    nt.assert_true(
        abs(numpy.mean(
            voltage[numpy.where((time < stop_time) & (time > start_time))]) -
            target_voltage) < precision)


@attr('bgscratch')
def test_search_hyp_current_replay_imap():
    """Tools: Test search_hyp_current_replay_imap"""
    blueconfig_location = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/"
    "simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/"
    "knockout/control/BlueConfig"
    gid_list = [107461, 107462]
    precision = .5
    target_voltage = -77
    start_time = 1
    stop_time = 5

    hyp_currents = {}
    results = bglibpy.tools.search_hyp_current_replay_imap(
        blueconfig_location,
        gid_list,
        timeout=150,
        target_voltage=target_voltage,
        min_current=-2.0,
        max_current=0.0,
        start_time=start_time,
        stop_time=stop_time,
        precision=precision,
        max_nestlevel=2,
        return_fullrange=False)

    unprocessed_gids = set(gid_list)
    for gid, result in results:
        if gid is None:
            break
        else:
            hyp_currents[gid] = result[0]
            unprocessed_gids.remove(gid)

    nt.assert_true(hyp_currents[107461] == -1.5)
    import math
    nt.assert_true(math.isnan(hyp_currents[107462]))
'''

def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bglibpy.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir,
        0)
    nt.assert_true(abs(SS_voltage - -73.9235504304) < 0.001)

    SS_voltage_stoch = bglibpy.calculate_SS_voltage_subprocess(
        "%s/examples/cell_example2/test_cell.hoc" % script_dir,
        "%s/examples/cell_example2" % script_dir,
        0)
    nt.assert_true(abs(SS_voltage_stoch - -73.9235504304) < 0.001)


class TestTools(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_empty" % script_dir)

    def test_holding_current(self):
        """Tools: Test holding_current"""

        holding_current, holding_voltage = bglibpy.tools.holding_current(
            -80, 1, 'BlueConfig')
        nt.assert_almost_equal(holding_current, -0.08019584734739738)
        nt.assert_almost_equal(holding_voltage, -80)

    def teardown(self):
        """Teardown"""
        os.chdir(self.prev_cwd)
