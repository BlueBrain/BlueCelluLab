from bluecellulab.simulation.parallel import IsolatedProcess


def test_isolated_process():
    """Test to ensure isolated process keeps its properties."""
    runner = IsolatedProcess()
    assert runner._processes == 1
    assert runner._maxtasksperchild == 1
