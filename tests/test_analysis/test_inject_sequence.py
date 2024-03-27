"""Unit tests for the inject_sequence module."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from bluecellulab import create_ball_stick
from bluecellulab.analysis.inject_sequence import StimulusName, apply_multiple_step_stimuli, run_stimulus
from bluecellulab.stimulus.factory import StimulusFactory


def test_run_stimulus():
    """Test the run_stimulus function."""
    template_params = create_ball_stick().template_params
    stimulus = StimulusFactory(dt=1.0).idrest(threshold_current=0.1)
    recording = run_stimulus(template_params, stimulus, "soma[0]", 0.5, 50.0)
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
    cell = create_ball_stick()

    with patch('bluecellulab.analysis.inject_sequence.IsolatedProcess') as mock_isolated_process, \
         patch('bluecellulab.analysis.inject_sequence.run_stimulus', mock_run_stimulus):
        # the mock process pool to return a list of MockRecordings
        mock_isolated_process.return_value.__enter__.return_value.starmap.return_value = [MockRecording() for _ in amplitudes]

        recordings = apply_multiple_step_stimuli(cell, StimulusName.FIRE_PATTERN, amplitudes, duration=400, n_processes=4)
        assert len(recordings) == len(amplitudes)
        for recording in recordings.values():
            assert len(recording.time) > 0
            assert len(recording.time) == len(recording.voltage)
            assert len(recording.time) == len(recording.current)

    # Testing unknown stimulus name
    with pytest.raises(ValueError) as exc_info:
        apply_multiple_step_stimuli(cell, "unknown", amplitudes, duration=400, n_processes=4)
    assert "Unknown stimulus name" in str(exc_info.value)

    short_amplitudes = [80]
    other_stim = [StimulusName.AP_WAVEFORM, StimulusName.IV, StimulusName.IDREST]
    for stim in other_stim:
        res = apply_multiple_step_stimuli(cell, stim, short_amplitudes, duration=4, n_processes=1)
        assert len(res) == len(short_amplitudes)
