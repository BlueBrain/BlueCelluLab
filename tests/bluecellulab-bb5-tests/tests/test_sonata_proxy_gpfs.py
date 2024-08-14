# Copyright 2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests the sonata proxy using a sim on gpfs."""

from pytest import approx, mark

from bluecellulab.cell import SonataProxy
from bluecellulab.circuit import BluepyCircuitAccess
from bluecellulab.circuit.node_id import CellId


test_relative_ornstein_path = (
    "/gpfs/bbp.cscs.ch/project/proj96/home/tuncel/simulations/"
    "test_sonata_proxy/BlueConfig"
)


class TestSonataProxy:
    def setup_method(self):
        circuit_access = BluepyCircuitAccess(test_relative_ornstein_path)
        cell_id = CellId("", 1)
        self.sonata_proxy = SonataProxy(cell_id, circuit_access)

    @mark.v6
    def test_get_input_resistance(self):
        assert self.sonata_proxy.get_input_resistance().iloc[0] == approx(262.087372)
