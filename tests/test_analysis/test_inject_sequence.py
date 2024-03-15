"""Unit tests for the inject_sequence module."""
import pytest
from bluecellulab import create_ball_stick
from bluecellulab.analysis.inject_sequence import StimulusName, apply_multiple_step_stimuli, run_stimulus
from bluecellulab.stimulus.factory import StimulusFactory


def test_run_stimulus():
    """Test the run_stimulus function."""
    template_params = create_ball_stick().template_params
    stimulus = StimulusFactory(dt=1.0).idrest(threshold_current=0.1)
    recording = run_stimulus(template_params, stimulus, None, 0.5, 50.0)
    assert len(recording.time) > 0
    assert len(recording.time) == len(recording.voltage)
    assert len(recording.time) == len(recording.current)


def test_apply_multiple_step_stimuli():
    """Test the apply_multiple_step_stimuli function."""
    cell = create_ball_stick()
    amplitudes = [80, 100, 120, 140]
    recordings = apply_multiple_step_stimuli(cell, StimulusName.FIRE_PATTERN, amplitudes, duration=40, n_processes=1)
    assert len(recordings) == len(amplitudes)
    for recording in recordings.values():
        assert len(recording.time) > 0
        assert len(recording.time) == len(recording.voltage)
        assert len(recording.time) == len(recording.current)

    with pytest.raises(ValueError) as exc_info:
        apply_multiple_step_stimuli(cell, "unknown", amplitudes, duration=400)
    assert "Unknown stimulus name" in str(exc_info.value)
