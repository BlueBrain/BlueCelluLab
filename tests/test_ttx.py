"""Unit tests for TTX in mod files"""

import os

import pytest

import bluecellulab

script_dir = os.path.dirname(__file__)


@pytest.mark.v5
def test_allNaChannels():
    """TTX: Testing ttx enabling"""

    na_channelnames = ['NaTs2_t']

    cell = bluecellulab.Cell(
        "%s/examples/cell_example_empty/test_cell.hoc" %
        script_dir,
        "%s/examples/cell_example_empty" %
        script_dir)

    for na_channelname in na_channelnames:
        cell.soma.insert(na_channelname)

        cell.add_step(0, 1000, .1)
        sim = bluecellulab.Simulation()
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
        assert voltage_nottx1[-1] != voltage_ttx[-1]

        assert voltage_nottx1[-1] == voltage_nottx2[-1]


@pytest.mark.v6
def test_allNaChannels_v6a():
    """TTX: Testing ttx enabling in v6a cell"""

    cell = bluecellulab.Cell(
        "%s/examples/cell_example_empty/test_cell_v6a.hoc" %
        script_dir,
        "%s/examples/cell_example_empty" %
        script_dir)

    cell.add_step(0, 1000, .1)
    sim = bluecellulab.Simulation()
    sim.add_cell(cell)

    sim.run(10)
    # time_nottx1 = cell.get_time()
    voltage_nottx1 = cell.get_soma_voltage()

    cell.enable_ttx()
    sim.run(10)
    # time_ttx = cell.get_time()
    voltage_ttx = cell.get_soma_voltage()

    cell.disable_ttx()
    sim.run(10)
    voltage_nottx2 = cell.get_soma_voltage()

    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(time_nottx1, voltage_nottx1, label='no ttx')
    plt.plot(time_ttx, voltage_ttx, label='ttx')
    plt.legend()
    plt.savefig('ttx.png')
    '''

    # Check if voltage changed due to enable_ttx

    assert voltage_nottx1[-1] != voltage_ttx[-1]
    assert voltage_nottx1[-1] == voltage_nottx2[-1]
