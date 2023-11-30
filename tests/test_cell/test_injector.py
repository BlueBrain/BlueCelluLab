"""Unit tests for the injector module."""

import math
from pathlib import Path

import numpy as np
from pydantic import ValidationError
import pytest
from pytest import approx, raises

import bluecellulab

from bluecellulab.circuit import SonataCircuitAccess
from bluecellulab.circuit.node_id import CellId
from bluecellulab.stimuli import (
    Pulse,
    Noise,
    Hyperpolarizing,
    RelativeLinear,
    OrnsteinUhlenbeck,
    RelativeOrnsteinUhlenbeck,
    ShotNoise,
    RelativeShotNoise,
    ClampMode,
)
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.cell.stimuli_generator import gen_shotnoise_signal
from bluecellulab.cell import SonataProxy

script_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v5
class TestInjector:
    """Test the InjectableMixin."""

    @classmethod
    def setup_method(cls):
        cls.cell = bluecellulab.Cell(f"{str(script_dir)}/examples/cell_example1/test_cell.hoc",
                                     f"{str(script_dir)}/examples/cell_example1/test_cell.asc")

        cls.sim = bluecellulab.Simulation()
        cls.sim.add_cell(cls.cell)

    def test_inject_pulse(self):
        """Test the pulse train injection."""
        stimulus = Pulse(
            target="single-cell",
            delay=2,
            duration=20,
            amp_start=4,
            frequency=5,
            width=2,
        )
        tstim = self.cell.add_pulse(stimulus)
        assert tstim.stim.to_python() == [0.0, 4.0, 4.0, 0.0, 0.0]
        assert tstim.tvec.to_python() == [2.0, 2.0, 4.0, 4.0, 22.0]

        with raises(ValidationError):
            unsupported_stimulus = Pulse(
                target="single-cell",
                delay=2,
                duration=20,
                amp_start=4,
                frequency=5,
                width=2,
                offset=1,
            )
            self.cell.add_pulse(unsupported_stimulus)

    def test_inject_step(self):
        """Test the step current injection."""
        tstim = self.cell.add_step(start_time=2.0, stop_time=6.0, level=1.0)
        assert tstim.stim.to_python() == [0.0, 1.0, 1.0, 0.0, 0.0]
        assert tstim.tvec.to_python() == [2.0, 2.0, 6.0, 6.0, 6.0]

    def test_inject_ramp(self):
        """Test the ramp injection."""
        tstim = self.cell.add_ramp(start_time=2.0, stop_time=6.0, start_level=0.5, stop_level=1)
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
            [66.5, -10.76796553, 17.8876999, 13.47384441, 12.09052411]), abs=1e-3)

    def test_get_noise_step_rand(self):
        """Unit test for _get_noise_step_rand."""
        noisestim_count = 5
        for mode in ["Compatibility", "UpdatedMCell", "Random123"]:
            rng_obj = bluecellulab.RNGSettings()
            rng_obj.mode = mode
            self.cell.rng_settings = rng_obj
            rng = self.cell._get_noise_step_rand(noisestim_count)
        rng.Random123()
        rng.poisson(4)
        rng.MCellRan4()
        assert rng.uniform(1, 15) == 2.9755221367813647

    def test_add_noise_step(self):
        """Test adding a step current with noise on top."""
        rng_obj = bluecellulab.RNGSettings()
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
        assert tstim.tvec.to_python() == [
            2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 0.0]

    def test_add_noise_step_with_seed(self):
        """Test adding a step current with noise on top with a seed."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Compatibility"
        tstim = self.cell.add_noise_step(
            section=self.cell.soma, segx=0.5, mean=2, variance=0.05, delay=2,
            duration=8, seed=1)

        assert tstim.stim.as_numpy() == approx(np.array(
            [0., 1.84104848, 1.97759473, 2.22855241, 1.80930735,
             2.09799701, 2.10379869, 2.29691643, 2.26258353, 2.14120033,
             1.93326057, 1.94724241, 1.87856356, 2.4008308, 1.91991524,
             1.50814262, 1.83374623, 0.]))

        assert tstim.tvec.to_python() == [
            2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 0.0]

    def test_add_replay_noise(self):
        """Unit test for add_replay_noise."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Compatibility"
        stimulus = Noise(
            mean_percent=1, variance=0.1, delay=4, duration=10, target="single-cell"
        )
        tstim = self.cell.add_replay_noise(stimulus, noise_seed=1)
        assert tstim.stim.as_numpy() == approx(np.array(
            [0., -0.00780348, 0.00048122, 0.01570763, -0.00972932,
             0.00778641, 0.00813842, 0.0198555, 0.01777241, 0.0104077,
             -0.00220868, -0.00136035, -0.00552732, 0.02616032, -0.00301838,
             -0.02800195, -0.00824653, -0.00273605, 0.00022639, 0.009682,
             0.00787559, 0.]), abs=1e-5)
        assert tstim.tvec.to_python() == [
            4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
            7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
            13.5, 14.0, 0.0]

    def test_add_replay_hypamp(self):
        """Unit test for add_replay_hypamp method."""
        stimulus = Hyperpolarizing(target="single-cell", delay=4, duration=20)
        tstim = self.cell.add_replay_hypamp(stimulus)
        hypamp = self.cell.hypamp
        assert tstim.stim.to_python() == [0.0, hypamp, hypamp, 0.0, 0.0]
        assert tstim.tvec.to_python() == [4.0, 4.0, 24.0, 24.0, 24.0]
        with pytest.raises(BluecellulabError):
            self.cell.hypamp = None
            self.cell.add_replay_hypamp(stimulus)

    def test_add_replay_relativelinear(self):
        """Unit test for add_replay_relativelinear."""
        stimulus = RelativeLinear(
            target="single-cell",
            delay=4, duration=20, percent_start=60)
        tstim = self.cell.add_replay_relativelinear(stimulus)
        assert tstim.stim.to_python() == [0.0, 0.1104372, 0.1104372, 0.0, 0.0]
        assert tstim.tvec.to_python() == [4.0, 4.0, 24.0, 24.0, 24.0]

    def test_get_ornstein_uhlenbeck_rand(self):
        """Unit test to check RNG generated for ornstein_uhlenbeck."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Random123"
        self.cell.rng_settings = rng_obj
        rng = self.cell._get_ornstein_uhlenbeck_rand(0, 144)
        assert rng.uniform(1, 15) == 12.477080945047298

        with raises(BluecellulabError):
            rng_obj.mode = "Compatibility"
            self.cell._get_ornstein_uhlenbeck_rand(0, 144)

    def test_get_shotnoise_step_rand(self):
        """Unit test to check RNG generated for shotnoise."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Random123"
        self.cell.rng_settings = rng_obj
        rng = self.cell._get_shotnoise_step_rand(0, 144)
        assert rng.uniform(1, 15) == 7.260484082563668

        with raises(BluecellulabError):
            rng_obj.mode = "Compatibility"
            self.cell._get_shotnoise_step_rand(0, 144)

    def test_add_replay_shotnoise(self):
        """Unit test for add_replay_shotnoise."""
        rng_obj = bluecellulab.RNGSettings(mode="Random123", base_seed=549821)
        rng_obj.stimulus_seed = 549821
        self.cell.rng_settings = rng_obj
        soma = self.cell.soma
        segx = 0.5
        stimulus = ShotNoise(
            target="single-cell", delay=0, duration=2,
            rise_time=0.4, decay_time=4, rate=2E3, amp_mean=40E-3, amp_var=16E-4,
            seed=3899663
        )
        time_vec, stim_vec = self.cell.add_replay_shotnoise(soma, segx, stimulus,
                                                            shotnoise_stim_count=3)
        assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                         1.75, 2.0, 2.0])
        assert list(stim_vec) == approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0077349976,
                                         0.0114066037, 0.0432062144, 0.0])
        with raises(ValidationError):
            invalid_stim = ShotNoise(
                target="single-cell", delay=0, duration=2,
                rise_time=4.4, decay_time=4, rate=2E3, amp_mean=40E-3, amp_var=16E-4,
                seed=3899663
            )
            self.cell.add_replay_shotnoise(soma, segx, invalid_stim, shotnoise_stim_count=3)

    def test_add_ornstein_uhlenbeck(self):
        """Unit test for add_ornstein_uhlenbeck."""
        rng_obj = bluecellulab.RNGSettings(mode="Random123", base_seed=549821)
        rng_obj.stimulus_seed = 549821
        self.cell.rng_settings = rng_obj
        soma = self.cell.soma
        segx = 0.5
        stimulus = OrnsteinUhlenbeck(
            target="single-cell", delay=0, duration=2,
            tau=2.8, sigma=0.0042, mean=0.029, mode="current_clamp", dt=0.25, seed=1
        )
        time_vec, stim_vec = self.cell.add_ornstein_uhlenbeck(soma, segx, stimulus,
                                                              stim_count=1)

        assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                         1.75, 2.0, 2.0])
        assert list(stim_vec) == approx([0.029, 0.030066357907250596, 0.0292377422585839,
                                         0.02740869851318549, 0.02615778315096319,
                                         0.026618755144424376, 0.026643574535378675,
                                         0.027504132283051937, 0.028251366180467998, 0.0])

        stimulus = OrnsteinUhlenbeck(
            target="single-cell", delay=0, duration=2,
            tau=2.8, sigma=0.0042, mean=0.029, mode="conductance", dt=0.25, seed=1
        )
        time_vec, stim_vec = self.cell.add_ornstein_uhlenbeck(soma, segx, stimulus,
                                                              stim_count=1)

        assert list(time_vec) == approx([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                         1.75, 2.0, 2.0])
        assert list(stim_vec) == approx([1000000000.0, 34.48275862068965, 33.259765053180814,
                                         34.20236730852261, 36.48476776520164, 38.22953933935253,
                                         37.56749684853171, 37.5325014544182, 36.35817300865007,
                                         35.39651829975447, 1000000000.0])

    def test_inject_current_clamp_signal(self):
        """Unit test for inject_current_clamp_signal."""
        tvec = bluecellulab.neuron.h.Vector(np.arange(10))
        svec = bluecellulab.neuron.h.Vector(np.random.normal(0, 0.1, 10))

        original_tvec = bluecellulab.neuron.h.Vector().copy(tvec)
        original_svec = bluecellulab.neuron.h.Vector().copy(svec)

        soma = self.cell.soma
        segx = 0.5
        self.cell.inject_current_clamp_signal(soma, segx, tvec, svec)

        assert len(self.cell.persistent) == 3
        assert "IClamp" in self.cell.persistent[0].hname()
        assert self.cell.persistent[0].dur == tvec[-1]  # check duration
        # tvec and svec are not modified
        assert original_tvec.eq(self.cell.persistent[1]) == 1.0
        assert original_svec.eq(self.cell.persistent[2]) == 1.0

    def test_inject_current_clamp_via_shotnoise_signal(self):
        """Unit test for inject_current_clamp_signal using a shotnoise_step."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Random123"
        self.cell.rng_settings = rng_obj

        soma = self.cell.soma
        segx = 0.5
        rng = self.cell._get_shotnoise_step_rand(shotnoise_stim_count=0, seed=None)
        tvec, svec = gen_shotnoise_signal(tau_D=4.0, tau_R=0.4, rate=2e3, amp_mean=40e-3,
                                          amp_var=16e-4, duration=2, dt=0.25, rng=rng)
        delay = 0
        tvec.add(delay)  # add delay
        tvec, svec = self.cell.inject_current_clamp_signal(soma, segx, tvec, svec)

        assert svec.as_numpy() == approx(np.array(
            [0., 0., 0., 0.00822223, 0.01212512,
             0.0137462, 0.034025, 0.04967694, 0.05614846, 0.]))
        assert tvec.to_python() == approx(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])

    def test_inject_dynamic_clamp_signal(self):
        """Unit test for inject_dynamic_clamp_signal."""
        tvec = bluecellulab.neuron.h.Vector(np.arange(10))
        svec = bluecellulab.neuron.h.Vector(np.random.normal(0, 0.1, 10))

        original_tvec = bluecellulab.neuron.h.Vector().copy(tvec)
        original_svec = bluecellulab.neuron.h.Vector().copy(svec)

        soma = self.cell.soma
        segx = 0.5
        reversal = 1e-3
        self.cell.inject_dynamic_clamp_signal(soma, segx, tvec, svec, reversal)
        assert len(self.cell.persistent) == 3
        assert "SEClamp" in self.cell.persistent[0].hname()
        assert self.cell.persistent[0].amp1 == reversal
        assert self.cell.persistent[0].dur1 == tvec[-1]  # check duration
        # tvec and svec are modified
        assert original_tvec.eq(self.cell.persistent[1]) != 1.0
        assert original_svec.eq(self.cell.persistent[2]) != 1.0

    def test_add_replay_relative_shotnoise(self):
        """Unit test for add_replay_relative_shotnoise."""
        rng_obj = bluecellulab.RNGSettings()
        rng_obj.mode = "Random123"
        self.cell.rng_settings = rng_obj
        stimulus = RelativeShotNoise(
            target="single-cell", delay=0, duration=2,
            rise_time=0.4, decay_time=4, mean_percent=70, sd_percent=40, amp_cv=0.63,
            seed=12,
        )
        self.cell.threshold = 0.184062
        soma = self.cell.soma
        segx = 0.5
        tvec, svec = self.cell.add_replay_relative_shotnoise(soma, segx, stimulus)
        assert svec.to_python() == approx([0., 0., 0., 0., 0.0204470197, 0.0301526984,
                                          0.0341840080, 0.0352485557, 0.0347913472, 0.])
        assert tvec.to_python() == approx([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                          2.0, 2.0])

        with raises(ValidationError):
            invalid_stim = RelativeShotNoise(
                target="single-cell", delay=0, duration=2,
                rise_time=4, decay_time=4, mean_percent=70, sd_percent=40, amp_cv=0.63,
                seed=12,
            )
            self.cell.add_replay_relative_shotnoise(soma, segx, invalid_stim)

        with raises(ZeroDivisionError):
            self.cell.threshold = 0.0
            self.cell.add_replay_relative_shotnoise(soma, segx, stimulus)

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


class TestInjectorSonata:
    """Test the InjectableMixin on local Sonata circuit."""

    @classmethod
    def setup_method(cls):
        circuit_path = (
            script_dir
            / "examples"
            / "sonata_unit_test_sims"
            / "condition_parameters"
            / "simulation_config.json"
        )
        circuit_access = SonataCircuitAccess(circuit_path)
        cell_id = CellId("NodeA", 1)
        hoc = circuit_access.emodel_path(cell_id)
        template_format = circuit_access.get_template_format()
        morph = circuit_access.morph_filepath(cell_id)
        emodel_properties = circuit_access.get_emodel_properties(cell_id)
        rng_settings = bluecellulab.RNGSettings(circuit_access=circuit_access)
        cls.cell = bluecellulab.Cell(template_path=hoc, morphology_path=morph, rng_settings=rng_settings,
                                     template_format=template_format, emodel_properties=emodel_properties)

        sonata_proxy = SonataProxy(cell_id, circuit_access)
        cls.cell.connect_to_circuit(sonata_proxy)

    @pytest.mark.v6
    def test_add_relative_ornstein_uhlenbeck_sd_error(self):
        """Unit test for add_relative_ornstein_uhlenbeck with 0 sd exception."""
        stimulus = RelativeOrnsteinUhlenbeck(
            target="single-cell", delay=0,
            duration=2, tau=2.8, mean_percent=3.078, sd_percent=0,
            mode=ClampMode.CURRENT, dt=0.25, seed=1
        )
        soma = self.cell.soma
        segx = 0.5
        with pytest.raises(BluecellulabError, match="standard deviation: 0.0, must be positive"):
            self.cell.add_relative_ornstein_uhlenbeck(soma, segx, stimulus)

    @pytest.mark.v6
    def test_add_relative_ornstein_uhlenbeck(self):
        """Unit test for adding relative ornstein_uhlenbeck."""
        stimulus = RelativeOrnsteinUhlenbeck(
            target="single-cell", delay=0,
            duration=2, tau=2.8, mean_percent=3.078, sd_percent=0.6156,
            mode=ClampMode.CURRENT, dt=0.25, seed=1
        )
        soma = self.cell.soma
        segx = 0.5
        time_vec, stim_vec = self.cell.add_relative_ornstein_uhlenbeck(
            soma, segx, stimulus)
        assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                         1.75, 2.0, 2.0])
        assert list(stim_vec) == approx(
            [0.050476, 0.048279, 0.049130, 0.050349, 0.049961, 0.054486,
             0.061342, 0.051148, 0.047710, 0.0], abs=1e-6)

    @pytest.mark.v6
    def test_add_alpha_synapse(self):
        """Unit test for adding alpha synapse stimulus to cell."""
        onset = 2.0
        tau = 10.0
        gmax = 0.0003
        e = 0.0
        section = self.cell.cell.getCell().dend[15]
        syn = self.cell.add_alpha_synapse(onset, tau, gmax, e, section)

        current_vector = bluecellulab.neuron.h.Vector()
        current_vector.record(syn._ref_i)
        sim = bluecellulab.Simulation()
        sim.add_cell(self.cell)
        sim.run(6.0, cvode=False, dt=1)
        current_vector = current_vector.to_python()
        # assert first 2 values are 0
        assert current_vector[0] == approx(0.0)
        assert current_vector[1] == approx(0.0)
        # assert last 3 values less than 0
        for j in range(-3, 0):
            assert current_vector[j] < 0.0
