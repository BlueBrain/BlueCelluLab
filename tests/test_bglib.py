# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for BGLib functionality"""

import os

import bluecellulab

script_dir = os.path.dirname(__file__)


class TestBGLibBaseClass1:

    """Test BGLib base class"""

    def setup_method(self):
        """Setup"""
        self.taper_cell = bluecellulab.Cell(
            "%s/examples/tapertest_cells/taper_cell.hoc" % script_dir,
            "%s/examples/tapertest_cells" % script_dir)
        self.notaper_cell = bluecellulab.Cell(
            "%s/examples/tapertest_cells/notaper_cell.hoc" % script_dir,
            "%s/examples/tapertest_cells" % script_dir)

    def teardown_method(self):
        """Teardown"""
        del self.taper_cell
        del self.notaper_cell

    def test_tapering(self):
        """BGLib: Test if tapering in Cell.hoc works correctly"""
        import numpy as np
        axon0_taper_diams = [segment.diam
                             for segment in self.taper_cell.axonal[0]]
        np.testing.assert_array_almost_equal(
            axon0_taper_diams, [
                1.9, 1.7, 1.5, 1.3, 1.1])

        axon1_taper_diams = [segment.diam
                             for segment in self.taper_cell.axonal[1]]
        np.testing.assert_array_almost_equal(
            axon1_taper_diams,
            [0.9642857142857143, 0.8928571428571429, 0.8214285714285714, 0.75,
             0.6785714285714286, 0.6071428571428571, 0.5357142857142857])

        myelin_taper_diams = [
            segment.diam
            for segment in self.taper_cell.cell.getCell().myelin[0]]
        np.testing.assert_array_almost_equal(
            myelin_taper_diams,
            [0.25] *
            21)

        axon0_notaper_diams = [segment.diam
                               for segment in self.notaper_cell.axonal[0]]
        np.testing.assert_array_almost_equal(
            axon0_notaper_diams,
            [0.7484678706370597])

        axon1_notaper_diams = [segment.diam
                               for segment in self.notaper_cell.axonal[1]]
        np.testing.assert_array_almost_equal(
            axon1_notaper_diams,
            [0.14187096846614167])
