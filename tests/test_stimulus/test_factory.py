import pytest
import numpy as np

from bluecellulab.stimulus.factory import Stimulus, Step, Ramp, StimulusFactory


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
