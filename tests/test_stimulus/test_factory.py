import pytest
import numpy as np

from bluecellulab.stimulus.factory import CombinedStimulus, EmptyStimulus, Stimulus, Step, Ramp, StimulusFactory


class TestStimulus:

    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            s = Stimulus(0.1)
            s.time
            s.current

    def test_repr(self):
        s = Step(0.1, 1, 2, 3)
        assert repr(s) == "Step(dt=0.1)"

    def test_len(self):
        s = Step(0.1, 1, 2, 3)
        assert len(s) == 10

    def test_plot(self):
        s = Step(0.1, 1, 2, 3)
        ax = s.plot()
        assert ax.get_xlabel() == "Time (ms)"
        assert ax.get_ylabel() == "Current (nA)"
        assert ax.get_title() == "Step"

    def test_plot_during_simulation(self):
        s = Step(0.1, 1, 2, 3)
        duration = 15
        ax = s.plot_during_simulation(duration)
        assert ax.get_xlim() == (0, duration)


class TestStimulusFactory:

    def setup_method(self):
        self.dt = 0.1
        self.factory = StimulusFactory(dt=self.dt)

    def test_create_step(self):
        s = self.factory.step(1, 2, 3)
        assert isinstance(s, Step)
        assert np.all(s.time == np.arange(1, 2, self.dt))
        assert np.all(s.current == np.full(10, 3))

    def test_create_ramp(self):
        s = self.factory.ramp(1, 2, 3, 4)
        assert isinstance(s, Ramp)
        assert np.all(s.time == np.arange(1, 2, self.dt))
        assert np.all(s.current == np.linspace(3, 4, 10))

    def test_create_ap_waveform(self):
        s = self.factory.ap_waveform(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        # below write np almost equal
        np.testing.assert_allclose(s.time, np.arange(250, 550, self.dt))
        assert s.current[0] == 2.2
        assert s.current[-1] == 0.0


def test_combined_stimulus():
    """Test combining Stimulus objects."""
    s1 = Step(0.1, 1, 2, 3)
    s2 = Step(0.1, 3, 4, 5)
    combined = s1 + s2

    assert isinstance(combined, CombinedStimulus)
    assert np.all(combined.time == np.concatenate([s1.time, s2.time + s1.time[-1] + 0.1]))
    assert np.all(combined.current == np.concatenate([s1.current, s2.current]))
    assert combined.dt == 0.1
    assert len(combined) == len(s1) + len(s2)
    assert combined.stimulus_time == (len(combined) * combined.dt)


def test_combined_stimulus_different_dt():
    s1 = Step(0.1, 1, 2, 3)
    s2 = Step(0.2, 3, 4, 5)
    with pytest.raises(ValueError):
        combined = s1 + s2


def test_empty_stimulus():
    dt = 0.2
    duration = 1.0
    stimulus = EmptyStimulus(dt, duration)

    assert stimulus.dt == dt
    assert np.all(stimulus.time == np.arange(0, duration, dt))
    assert np.all(stimulus.current == np.zeros_like(stimulus.time))
