"""Unit tests for TTX in mod files"""

# pylint: disable=

import os

script_dir = os.path.dirname(__file__)

import nose.tools as nt

import bglibpy


def test_allNaChannels():
    """TTX: Testing ttx enabling"""

    na_channelnames = ['Na']

    cell = bglibpy.Cell(
        "%s/examples/cell_example_empty/test_cell.hoc" %
        script_dir,
        "%s/examples/cell_example_empty" %
        script_dir)

    for na_channelname in na_channelnames:
        cell.soma.insert(na_channelname)

        cell.add_step(0, 1000, .1)
        sim = bglibpy.Simulation()
        sim.add_cell(cell)

        sim.run(10)
        voltage_nottx1 = cell.get_soma_voltage()

        cell.enable_ttx()
        sim.run(10)
        voltage_ttx = cell.get_soma_voltage()

        cell.disable_ttx()
        sim.run(10)
        voltage_nottx2 = cell.get_soma_voltage()

        # Check if voltage changed due to enable_ttx
        nt.assert_not_equal(voltage_nottx1[-1], voltage_ttx[-1])

        nt.assert_equal(voltage_nottx1[-1], voltage_nottx2[-1])
