import pytest
import numpy as np

from bluecellulab.stimulus.factory import CombinedStimulus, EmptyStimulus, Stimulus, Step, Ramp, StimulusFactory


class TestStimulus:

    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            s = Stimulus(0.1)

    def test_repr(self):
        s = Step.amplitude_based(0.1, 1, 2, 3, 0.55)
        assert repr(s) == "CombinedStimulus(dt=0.1)"

    def test_len(self):
        s = Step.amplitude_based(0.1, 0, 1, 0, 0.55)
        assert len(s) == 10

    def test_plot(self):
        s = Step.amplitude_based(0.1, 1, 2, 3, 0.55)
        ax = s.plot()
        assert ax.get_xlabel() == "Time (ms)"
        assert ax.get_ylabel() == "Current (nA)"
        assert ax.get_title() == "CombinedStimulus"


class TestStimulusFactory:

    def setup_method(self):
        self.dt = 0.1
        self.factory = StimulusFactory(dt=self.dt)

    def test_create_step(self):
        stim = self.factory.step(0, 1, 0, 0.55)
        assert isinstance(stim, CombinedStimulus)
        assert np.all(stim.time == np.arange(0, 1, self.dt))
        assert np.all(stim.current == np.full(10, 0.55))

    def test_create_ramp(self):
        stim = self.factory.ramp(1, 2, 0, amplitude=3)
        assert isinstance(stim, CombinedStimulus)
        np.testing.assert_almost_equal(stim.time, np.arange(0, 2, self.dt), decimal=9)
        assert stim.current[0] == 0.0
        assert stim.current[-1] == 3.0

    def test_create_ap_waveform(self):
        s = self.factory.ap_waveform(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        # below write np almost equal
        np.testing.assert_allclose(s.time, np.arange(0, 550, self.dt))
        assert s.current[0] == 0.0
        assert s.current[2500] == 2.2
        assert s.current[-1] == 0.0

    def test_create_idrest(self):
        s = self.factory.idrest(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 1850

    def test_create_iv(self):
        s = self.factory.iv(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 3500
        # assert there are negative values
        assert np.any(s.current < 0)
        # assert no positive values
        assert not np.any(s.current > 0)

    def test_create_fire_pattern(self):
        s = self.factory.fire_pattern(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 4100


def test_combined_stimulus():
    """Test combining Stimulus objects."""
    s1 = Step.amplitude_based(0.1, 0.55, 1, 2, 3)
    s2 = Step.threshold_based(0.1, 0.55, 200, 3, 4, 5)
    combined = s1 + s2

    assert isinstance(combined, CombinedStimulus)
    assert np.all(combined.time == np.concatenate([s1.time, s2.time + s1.time[-1] + 0.1]))
    assert np.all(combined.current == np.concatenate([s1.current, s2.current]))
    assert combined.dt == 0.1
    assert len(combined) == len(s1) + len(s2)
    assert combined.stimulus_time == (len(combined) * combined.dt)


def test_combined_stimulus_different_dt():
    s1 = Step.amplitude_based(0.1, 0.55, 1, 2, 3)
    s2 = Step.threshold_based(0.2, 0.55, 200, 3, 4, 5)
    with pytest.raises(ValueError):
        combined = s1 + s2


def test_empty_stimulus():
    dt = 0.2
    duration = 1.0
    stimulus = EmptyStimulus(dt, duration)

    assert stimulus.dt == dt
    assert np.all(stimulus.time == np.arange(0, duration, dt))
    assert np.all(stimulus.current == np.zeros_like(stimulus.time))


def test_combine_multiple_stimuli():
    """Test combining multiple stimuli."""
    dt = 0.1
    stim1 = Step.amplitude_based(dt, 50, 100, 50, 0.55)
    stim2 = Ramp.amplitude_based(dt, 3, 4, 2, 0.55)
    stim3 = EmptyStimulus(dt, 1)
    stim4 = Step.amplitude_based(dt, 5, 10, 5, 0.66)

    combined = stim1 + stim2 + stim3 + stim4

    assert isinstance(combined, CombinedStimulus)
    assert combined.dt == dt

    shifted_stim2_time = stim2.time + stim1.time[-1] + dt
    shifted_stim3_time = stim3.time + shifted_stim2_time[-1] + dt
    shifted_stim4_time = stim4.time + shifted_stim3_time[-1] + dt

    expected_time = np.concatenate([
        stim1.time,
        shifted_stim2_time,
        shifted_stim3_time,
        shifted_stim4_time,
    ])
    expected_current = np.concatenate([
        stim1.current,
        stim2.current,
        stim3.current,
        stim4.current,
    ])

    assert np.all(combined.time == expected_time)
    assert np.all(combined.current == expected_current)
