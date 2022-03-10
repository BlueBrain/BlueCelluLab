"""Unit tests for the injector module."""

import os
import math

import numpy as np
from pytest import approx, raises, mark

import bglibpy
from bglibpy.exceptions import BGLibPyError

script_dir = os.path.dirname(__file__)


class TestInjector:
    """Test the InjectMixin."""

    @classmethod
    def setup_method(self):
        self.cell = bglibpy.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % script_dir,
            "%s/examples/cell_example1/test_cell.asc" % script_dir)
        self.sim = bglibpy.Simulation()
        self.sim.add_cell(self.cell)

    def test_inject_pulse(self):
        """Test the pulse train injection."""
        stimulus = {
            "Delay": 2, "Duration": 20, "AmpStart": 4,"Frequency": 5, "Width": 2}
        tstim = self.cell.add_pulse(stimulus)
        assert tstim.stim.to_python() == [0.0, 4.0, 4.0, 0.0, 0.0]
        assert tstim.tvec.to_python() == [2.0, 2.0, 4.0, 4.0, 22.0]

        with raises(BGLibPyError):
            unsupported_stimulus = {"Pattern": "pattern1", "Offset": 1}
            self.cell.add_pulse(unsupported_stimulus)

    def test_inject_step(self):
        """Test the step current injection."""
        tstim = self.cell.add_step(start_time=2.0, stop_time=6.0, level=1.0)
        assert tstim.stim.to_python() == [0.0, 1.0, 1.0, 0.0, 0.0]
        assert tstim.tvec.to_python() == [2.0, 2.0, 6.0, 6.0, 6.0]

    def test_inject_ramp(self):
        """Test the ramp injection."""
        tstim = self.cell.add_ramp(start_time=2.0, stop_time=6.0, start_level= 0.5, stop_level=1)
        assert tstim.stim.to_python() == [0.0, 0.0, 0.5, 1.0, 0.0, 0.0]
        assert tstim.tvec.to_python() == [0.0, 2.0, 2.0, 6.0, 6.0, 6.0]

    def test_voltage_clamp(self):
        """Test adding voltage clamp."""
        amp = 1.5
        stop_time = 10
        rs = 1
        n_recording = "test_volt_clamp"
        seclamp_obj = self.cell.add_voltage_clamp(
            stop_time=stop_time, level=amp, rs=rs,
            current_record_name=n_recording
            )
        assert seclamp_obj.amp1 == amp
        assert seclamp_obj.dur1 == stop_time
        assert seclamp_obj.rs == rs

        self.sim.run(10, dt=1, cvode=False)
        current = self.cell.get_recording("test_volt_clamp")
        assert current == approx(np.array(
            [66.5, 5.39520998, -10.76796553, 20.6887735,
             17.8876999, 15.14995787, 13.47384441, 12.55945316,
             12.09052411, 11.8250991, 11.5502658]), abs=1e-3)

    def test_voltage_clamp_dt(self):
        """Test adding voltage clamp to a cell with a dt value."""
        amp = 1.5
        stop_time = 10
        rs = 1
        n_recording = "test_volt_clamp_dt"
        self.cell.record_dt = 2
        seclamp_obj = self.cell.add_voltage_clamp(
            stop_time=stop_time, level=amp, rs=rs,
            current_record_name=n_recording
            )
        assert seclamp_obj.amp1 == amp
        assert seclamp_obj.dur1 == stop_time
        assert seclamp_obj.rs == rs

        self.sim.run(10, dt=1, cvode=False)
        current = self.cell.get_recording("test_volt_clamp_dt")
        assert current == approx(np.array(
            [ 66.5, -10.76796553, 17.8876999, 13.47384441, 12.09052411]), abs=1e-3)

    def test_get_noise_step_rand(self):
        """Unit test for _get_noise_step_rand."""
        noisestim_count = 5
        for mode in ["Compatibility", "UpdatedMCell", "Random123"]:
            rng_obj = bglibpy.RNGSettings()
            rng_obj.mode = mode
            self.cell.rng_settings = rng_obj
            rng = self.cell._get_noise_step_rand(noisestim_count)
        rng.Random123()
        rng.poisson(4)
        rng.MCellRan4()
        assert rng.uniform(1,15) == 2.9755221367813647

    def test_add_noise_step(self):
        """Test adding a step current with noise on top."""
        rng_obj = bglibpy.RNGSettings()
        rng_obj.mode = "Compatibility"
        self.cell.rng_settings = rng_obj
        tstim = self.cell.add_noise_step(
            section=self.cell.soma, segx=0.5, mean=2, variance=0.05, delay=2,
            duration=8, noisestim_count=5)
        assert tstim.stim.as_numpy() == approx(np.array(
            [0.0, 1.688244150056253,
             1.5206057728916358, 1.963561732554084, 1.5287573262092622,
             1.8136627185053698, 2.1230204494073135, 1.8715777361739463,
             1.7068988305615118, 1.7574514888132944, 2.055318487170783,
             1.8673307717912755, 1.932569903725156, 1.9394341839268754,
             1.8843667144133713, 1.8175816051992186, 1.927545675194812, 0.0]))
        assert tstim.tvec.to_python() == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 0.0]

    def test_add_noise_step_with_seed(self):
        """Test adding a step current with noise on top with a seed."""
        rng_obj = bglibpy.RNGSettings()
        rng_obj.mode = "Compatibility"
        tstim = self.cell.add_noise_step(
            section=self.cell.soma, segx=0.5, mean=2, variance=0.05, delay=2,
            duration=8, seed=1)

        assert tstim.stim.as_numpy() == approx(np.array(
            [0., 1.84104848, 1.97759473, 2.22855241, 1.80930735,
             2.09799701, 2.10379869, 2.29691643, 2.26258353, 2.14120033,
             1.93326057, 1.94724241, 1.87856356, 2.4008308 , 1.91991524,
             1.50814262, 1.83374623, 0.]))

        assert tstim.tvec.to_python() == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
         5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 0.0]

    def test_add_replay_noise(self):
        """Unit test for add_replay_noise."""
        rng_obj = bglibpy.RNGSettings()
        rng_obj.mode = "Compatibility"
        stimulus = {
            "MeanPercent": 1, "Variance": 0.1, "Delay": 4,"Duration": 10}
        tstim = self.cell.add_replay_noise(stimulus, noise_seed=1)
        assert tstim.stim.as_numpy() == approx(np.array(
            [0., -0.00780348, 0.00048122, 0.01570763, -0.00972932,
             0.00778641, 0.00813842, 0.0198555, 0.01777241, 0.0104077,
             -0.00220868, -0.00136035, -0.00552732, 0.02616032, -0.00301838,
             -0.02800195, -0.00824653, -0.00273605, 0.00022639, 0.009682,
             0.00787559, 0.]), abs=1e-5)
        assert tstim.tvec.to_python() == [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
            7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
            13.5, 14.0, 0.0]

    def test_add_replay_hypamp(self):
        """Unit test for add_replay_hypamp method."""
        stimulus = {"Delay": 4, "Duration": 20}
        tstim = self.cell.add_replay_hypamp(stimulus)
        hypamp = self.cell.hypamp
        assert tstim.stim.to_python() == [0.0, hypamp, hypamp, 0.0, 0.0]
        assert tstim.tvec.to_python() == [4.0, 4.0, 24.0, 24.0, 24.0]

    def test_add_replay_relativelinear(self):
        """Unit test for add_replay_relativelinear."""
        stimulus = {"Delay": 4, "Duration": 20, "PercentStart": 60}
        tstim = self.cell.add_replay_relativelinear(stimulus)
        assert tstim.stim.to_python() == [0.0, 0.1104372, 0.1104372, 0.0, 0.0]
        assert tstim.tvec.to_python() == [4.0, 4.0, 24.0, 24.0, 24.0]

    def test_inject_current_waveform(self):
        """Test injecting any input current and time arrays."""
        start_time, stop_time, dt = 10.0, 20.0, 1.0
        amplitude, freq, mid_level = 12.0, 50.0, 0.0

        t_content = np.arange(start_time, stop_time, dt)
        i_content = [amplitude * math.sin(freq * (x - start_time) * (
            2 * math.pi)) + mid_level for x in t_content]

        current = self.cell.injectCurrentWaveform(t_content, i_content)
        assert current.as_numpy() == approx(np.array(
            [0.00000000e+00, 2.35726407e-14, 4.71452814e-14, -6.11403104e-13,
             9.42905627e-14, -1.92849988e-12, -1.22280621e-12, -3.24559665e-12,
             1.88581125e-13, -1.83420931e-12]), abs=1e-3)

    def test_add_sin_current(self):
        """Unit test for add_sin_current."""
        start_time, duration = 2, 10
        amp, freq = 12.0, 50.0
        tstim = self.cell.add_sin_current(amp, start_time, duration, freq)
        assert tstim.stim.as_numpy()[195:205] == approx(np.array(
            [11.99074843, 11.99407872, 11.99666916, 11.99851959, 11.99962989,
             12., 11.99962989, 11.99851959, 11.99666916, 11.99407872]))
