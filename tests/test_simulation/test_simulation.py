"""Tests for the Simulation class"""

import pathlib

import numpy as np
from pytest import approx
import pytest

import bluecellulab
from bluecellulab.simulation.neuron_globals import set_temperature

parent_dir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.v5
class TestCellBaseClass:
    """Base test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            parent_dir / "examples/cell_example1/test_cell.hoc",
            str(parent_dir / "examples/cell_example1"))
        self.sim = bluecellulab.Simulation()
        self.sim.add_cell(self.cell)
        assert isinstance(self.sim, bluecellulab.Simulation)

    def teardown_method(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run the simulation for 20 ms"""
        self.sim.run(20)


@pytest.mark.v5
class TestCellcSTUTBaseClass:

    """Base test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            parent_dir / "examples/cell_example_cstut/cSTUT_7.hoc",
            str(parent_dir / "examples/cell_example_cstut"))
        self.sim = bluecellulab.Simulation()
        self.sim.add_cell(self.cell)
        assert isinstance(self.sim, bluecellulab.Simulation)

    def teardown_method(self):
        """Destructor"""
        del self.cell
        del self.sim

    def test_run(self):
        """Simulation: Run a simulation with cSTUT and MCellRan4 for 300 ms"""
        set_temperature(34.0)
        self.sim.run(300, cvode=False)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[np.where(time > 150)]

        # Lowered precision because of
        # commit 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn repo
        assert np.mean(voltage_ss) == approx(-75.61918061202924, abs=1e-6)
        assert np.std(voltage_ss) == approx(0.19192736450671288, abs=1e-6)
