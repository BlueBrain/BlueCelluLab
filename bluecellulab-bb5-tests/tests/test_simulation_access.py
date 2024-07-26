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
from pathlib import Path

import numpy as np
import pytest

from bluecellulab.circuit import BluepySimulationAccess, CellId

from helpers.circuit import blueconfig_append_path


parent_dir = Path(__file__).resolve().parent


class TestBluepySimulationAccess:
    def setup_method(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"
        modified_conf = blueconfig_append_path(
            conf_pre_path / "BlueConfig", conf_pre_path
        )
        self.simulation_access = BluepySimulationAccess(modified_conf)

    def test_init_file_not_found(self):
        """Test BluepySimulationAccess init edge cases."""
        with pytest.raises(FileNotFoundError):
            BluepySimulationAccess(Path("non_existing_file"))

    def test_get_soma_voltage(self):
        """Test BluepySimulationAccess.get_soma_voltage."""
        cell_id = CellId("", 1)
        t_start, t_end, t_step = 0, 50, 0.1
        soma_voltage = self.simulation_access.get_soma_voltage(
            cell_id, t_start, t_end, t_step
        )
        assert soma_voltage.shape == (int((t_end - t_start) / t_step),)
        assert soma_voltage[0] == -65.0
        assert np.mean(soma_voltage) == pytest.approx(-57.15379)
        assert np.max(soma_voltage) == pytest.approx(24.699245)
        assert np.min(soma_voltage) == pytest.approx(-65.15369)

    def test_get_soma_time_trace(self):
        """Test BluepySimulationAccess.get_soma_time_trace."""
        t_step = 0.1
        soma_time_trace = self.simulation_access.get_soma_time_trace(t_step)
        assert soma_time_trace.shape == (1000,)
        assert soma_time_trace[0] == 0.0
        assert soma_time_trace[1] == 0.1
        assert soma_time_trace[-1] == 99.9

    def test_get_spikes(self):
        """Test BluepySimulationAccess.get_spikes."""
        spikes = self.simulation_access.get_spikes()
        assert len(spikes[("", 2)]) == 7
        assert all(spikes[("", 2)][0:4] == np.array([15.0, 30.0, 45.0, 60.0]))
        assert spikes[("", 2)][-1] == 5000000.0
