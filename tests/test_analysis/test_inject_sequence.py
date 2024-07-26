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
"""Unit tests for the inject_sequence module."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from bluecellulab import create_ball_stick
from bluecellulab.analysis.inject_sequence import StimulusName, apply_multiple_stimuli, run_stimulus
from bluecellulab.stimulus.factory import StimulusFactory


def test_run_stimulus():
    """Test the run_stimulus function."""
    template_params = create_ball_stick().template_params
    stimulus = StimulusFactory(dt=1.0).idrest(threshold_current=0.1)
    recording = run_stimulus(template_params, stimulus, "soma[0]", 0.5)
    assert len(recording.time) > 0
    assert len(recording.time) == len(recording.voltage)
    assert len(recording.time) == len(recording.current)


class MockRecording:
    def __init__(self):
        self.time = [1, 2, 3]
        self.voltage = [-70, -55, -40]
        self.current = [0.1, 0.2, 0.3]


@pytest.fixture
def mock_run_stimulus():
    return MagicMock(return_value=MockRecording())


def test_apply_multiple_step_stimuli(mock_run_stimulus):
    """Do not run the code in parallel, mock the return value via MockRecording."""
    amplitudes = [80, 100, 120, 140]
    thres_perc = [0.08]
    cell = create_ball_stick()

    with patch('bluecellulab.analysis.inject_sequence.IsolatedProcess') as mock_isolated_process, \
         patch('bluecellulab.analysis.inject_sequence.run_stimulus', mock_run_stimulus):
        # the mock process pool to return a list of MockRecordings
        mock_isolated_process.return_value.__enter__.return_value.starmap.return_value = [MockRecording() for _ in amplitudes]

        recordings = apply_multiple_stimuli(cell, StimulusName.FIRE_PATTERN, amplitudes, threshold_based=False, n_processes=4)
        recordings_thres = apply_multiple_stimuli(cell, StimulusName.FIRE_PATTERN, thres_perc, n_processes=4)
        assert len(recordings) == len(amplitudes)
        assert len(recordings_thres) == len(thres_perc)
        for recording in list(recordings.values()) + list(recordings_thres.values()):
            assert len(recording.time) > 0
            assert len(recording.time) == len(recording.voltage)
            assert len(recording.time) == len(recording.current)

    # Testing unknown stimulus name
    with pytest.raises(ValueError) as exc_info:
        apply_multiple_stimuli(cell, "unknown", amplitudes, n_processes=4)
    assert "Unknown stimulus name" in str(exc_info.value)

    short_amplitudes = [80]
    short_thres = [0.08]
    other_stim = [StimulusName.AP_WAVEFORM, StimulusName.IV, StimulusName.IDREST, StimulusName.POS_CHEOPS, StimulusName.NEG_CHEOPS]
    for stim in other_stim:
        res = apply_multiple_stimuli(cell, stim, short_amplitudes, threshold_based=False, n_processes=1)
        res_thres = apply_multiple_stimuli(cell, stim, short_thres, n_processes=1)
        assert len(res) == len(short_amplitudes)
        assert len(res_thres) == len(short_thres)
