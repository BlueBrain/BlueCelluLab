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
import os

from pytest import approx
import pytest

from bluecellulab.tools import holding_current

from helpers.circuit import blueconfig_append_path

script_dir = os.path.dirname(__file__)


class TestTools:
    """Class to test SSim with two cell circuit"""

    def setup_method(self):
        """Setup"""
        conf_pre_path = os.path.join(script_dir, "examples", "sim_twocell_empty")
        blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

        self.config = blueconfig_append_path(blueconfig_path, conf_pre_path)

    @pytest.mark.v5
    def test_holding_current(self):
        """Tools: Test holding_current"""

        gid = 1
        expected_voltage = -80
        for expected_current, ttx in [
            (-0.08019584734739738, False),
            (-0.08019289690395226, True),
        ]:
            current, voltage = holding_current(
                expected_voltage, gid, self.config, enable_ttx=ttx
            )
            current == approx(expected_current, abs=1e-6)
            voltage == approx(expected_voltage, abs=1e-6)
