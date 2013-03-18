#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for tools.py"""

import nose.tools as nt
import bglibpy
import multiprocessing

def test_search_hyp_current_replay_gidlist():
    """Tools: Test search_hyp_current_replay_gidlist"""
    blueconfig_location = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"
    #gids = [107462, 107461]
    gid = 107461
    precision = .5
    target_voltage = -77
    start_time = 1
    stop_time = 5

    results = bglibpy.search_hyp_current_replay_gidlist(blueconfig_location, [gid],
        target_voltage=target_voltage,
        min_current=-2.0,
        max_current=0.0,
        start_time=start_time,
        stop_time=stop_time,
        precision=precision,
        max_nestlevel=5,
        return_fullrange=False
        )

    nt.assert_true(gid in results)
    step_level, (time, voltage) = results[gid]
    nt.assert_equal(step_level, -1.5)
    import numpy
    nt.assert_true(abs(numpy.mean(voltage[numpy.where((time < stop_time) & (time > start_time))])-target_voltage) < precision)

def test_search_hyp_current_replay_imap():
    """Tools: Test search_hyp_current_replay_gidlist"""
    blueconfig_location = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"
    gid_list = [107461]
    #gid_list = [107461, 107462]
    #gid = 107461
    precision = .5
    target_voltage = -77
    start_time = 1
    stop_time = 5

    print "Starting"
    bglibpy.tools.VERBOSE_LEVEL = 10
    results = bglibpy.tools.search_hyp_current_replay_imap(blueconfig_location, gid_list,
        target_voltage=target_voltage,
        min_current=-2.0,
        max_current=0.0,
        start_time=start_time,
        stop_time=stop_time,
        precision=precision,
        max_nestlevel=1,
        return_fullrange=False
        )
    level_traces = {}
    unprocessed_gids = set(gid_list)
    for _ in gid_list:
        try:
            if len(unprocessed_gids) == 1:
                timeout = 1000
            else:
                timeout = 1000
            (gid, result) = results.next(timeout=timeout)
            level_traces[gid] = result
            #Dosomething with gid and result (like save it to a file)
            unprocessed_gids.remove(gid)
        except StopIteration:
            break
        except multiprocessing.TimeoutError:
            pass
    print "Unprocessed gids: %s" % str(list(unprocessed_gids))

    print level_traces
    #for gid in gid_list:
    #    nt.assert_true(gid in level_traces)
    #step_level, (time, voltage) = results[gid]
    #nt.assert_equal(step_level, -1.5)
    #import numpy
    #nt.assert_true(abs(numpy.mean(voltage[numpy.where((time < stop_time) & (time > start_time))])-target_voltage) < precision)

def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bglibpy.calculate_SS_voltage_subprocess("examples/cell_example1/test_cell.hoc", "examples/cell_example1", 0)
    nt.assert_true(abs(SS_voltage - -73.9235504304) < 0.001)

    SS_voltage_stoch = bglibpy.calculate_SS_voltage_subprocess("examples/cell_example2/test_cell.hoc", "examples/cell_example2", 0)
    nt.assert_true(abs(SS_voltage_stoch - -73.9235504304) < 0.001)
