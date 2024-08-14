# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Simulation class"""

import pathlib

import numpy as np
from pytest import approx
import pytest

import bluecellulab

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
        self.sim.run(300, cvode=False)

        time = self.cell.get_time()
        voltage = self.cell.get_soma_voltage()

        voltage_ss = voltage[np.where(time > 150)]

        # Lowered precision because of
        # commit 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn repo
        assert np.mean(voltage_ss) == approx(-75.61918061202924, abs=1e-6)
        assert np.std(voltage_ss) == approx(0.19192736450671288, abs=1e-6)
