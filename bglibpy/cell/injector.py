"""Contains injection functionality for the cell."""

import math
import warnings

import numpy as np

import bglibpy
from bglibpy import BGLibPyError, ConfigError, lazy_printv, tools
from bglibpy.cell.stimuli_generator import (
    gen_ornstein_uhlenbeck,
    gen_shotnoise_signal,
    get_relative_shotnoise_params,
)


class InjectableMixin:
    """Mixin responsible of injections to the cell.
       Important Usage Note: Adds the instantiated Neuron objects to
        self.persistent to explicitly destroy them when their lifetime ends.
    """

    def relativity_proportion(self, stim_mode: str) -> float:
        """Relativity proportion used in Relative stimuli e.g. relative shotnoise."""
        if stim_mode == "Conductance":
            input_resistance = self.sonata_proxy.get_input_resistance().iloc[0]  # type: ignore
            return 1.0 / input_resistance
        else:  # Current
            return self.threshold  # type: ignore

    def add_pulse(self, stimulus):
        """Inject pulse stimulus for replay."""
        tstim = bglibpy.neuron.h.TStim(0.5, sec=self.soma)
        if 'Offset' in stimulus.keys():
            raise BGLibPyError("Found stimulus with pattern %s and Offset, "
                               "not supported" % stimulus['Pattern'])
        else:
            delay = float(stimulus['Delay'])

        tstim.train(delay,
                    float(stimulus['Duration']),
                    float(stimulus['AmpStart']),
                    float(stimulus['Frequency']),
                    float(stimulus['Width']))
        self.persistent.append(tstim)
        return tstim

    def add_step(self, start_time, stop_time, level, section=None, segx=0.5):
        """Add a step current injection."""
        if section is None:
            section = self.soma

        tstim = bglibpy.neuron.h.TStim(segx, sec=section)
        duration = stop_time - start_time
        tstim.pulse(start_time, duration, level)
        self.persistent.append(tstim)
        return tstim

    def add_ramp(self, start_time, stop_time, start_level, stop_level,
                 section=None, segx=0.5):
        """Add a ramp current injection."""
        if section is None:
            section = self.soma

        tstim = bglibpy.neuron.h.TStim(segx, sec=section)

        tstim.ramp(
            0.0,
            start_time,
            start_level,
            stop_level,
            stop_time - start_time,
            0.0,
            0.0)

        self.persistent.append(tstim)
        return tstim

    def add_voltage_clamp(
            self, stop_time, level, rs=None, section=None, segx=0.5,
            current_record_name=None, current_record_dt=None):
        """Add a voltage clamp

        Parameters
        ----------

        stop_time : float
            Time at which voltage clamp should stop
        level : float
            Voltage level of the vc (in mV)
        rs: float
            Series resistance of the vc (in MOhm)
        section: NEURON object
            Object representing the section to place the vc
        segx: float
            Segment x coordinate to place the vc
        current_record_name: str
            Name of the recording that will store the current
        current_record_dt: float
            Timestep to use for the recording of the current

        Returns
        -------

        SEClamp (NEURON) object of the created vc

        """

        if section is None:
            section = self.soma
        if current_record_dt is None:
            current_record_dt = self.record_dt
        vclamp = bglibpy.neuron.h.SEClamp(segx, sec=section)
        self.persistent.append(vclamp)

        vclamp.amp1 = level
        vclamp.dur1 = stop_time

        if rs is not None:
            vclamp.rs = rs

        current = bglibpy.neuron.h.Vector()
        if current_record_dt is None:
            current.record(vclamp._ref_i)
        else:
            current.record(
                vclamp._ref_i,
                self.get_precise_record_dt(current_record_dt))

        self.recordings[current_record_name] = current
        return vclamp

    def _get_noise_step_rand(self, noisestim_count):
        """Return rng for noise step stimulus"""
        if self.rng_settings.mode == "Compatibility":
            rng = bglibpy.neuron.h.Random(self.gid + noisestim_count)
        elif self.rng_settings.mode == "UpdatedMCell":
            rng = bglibpy.neuron.h.Random()
            rng.MCellRan4(
                noisestim_count * 10000 + 100,
                self.rng_settings.base_seed +
                self.rng_settings.stimulus_seed +
                self.gid * 1000)
        elif self.rng_settings.mode == "Random123":
            rng = bglibpy.neuron.h.Random()
            rng.Random123(
                noisestim_count + 100,
                self.rng_settings.stimulus_seed + 500,
                self.gid + 300)

        self.persistent.append(rng)
        return rng

    def add_noise_step(self, section,
                       segx,
                       mean, variance,
                       delay,
                       duration, seed=None, noisestim_count=None):
        """Inject a step current with noise on top."""
        if seed is not None:
            rand = bglibpy.neuron.h.Random(seed)
        else:
            rand = self._get_noise_step_rand(noisestim_count)

        tstim = bglibpy.neuron.h.TStim(segx, rand, sec=section)
        tstim.noise(delay, duration, mean, variance)
        self.persistent.append(rand)
        self.persistent.append(tstim)
        return tstim

    def add_replay_noise(
            self,
            stimulus,
            noise_seed=None,
            noisestim_count=None):
        """Add a replay noise stimulus."""
        mean = (float(stimulus['MeanPercent']) * self.threshold) / 100.0
        variance = (float(stimulus['Variance']) * self.threshold) / 100.0
        delay = float(stimulus['Delay'])
        duration = float(stimulus['Duration'])
        tstim = self.add_noise_step(
            self.soma,
            0.5,
            mean,
            variance,
            delay,
            duration,
            seed=noise_seed,
            noisestim_count=noisestim_count)

        lazy_printv("Added noise stimulus to gid %d: "
                    "delay=%f, duration=%f, mean=%f, variance=%f" %
                    (self.gid, delay, duration, mean, variance), 50)
        return tstim

    def add_replay_hypamp(self, stimulus):
        """Inject hypamp for the replay."""
        tstim = bglibpy.neuron.h.TStim(0.5, sec=self.soma)
        delay = float(stimulus['Delay'])
        duration = float(stimulus['Duration'])
        amp = self.hypamp
        tstim.pulse(delay, duration, amp)
        self.persistent.append(tstim)
        lazy_printv("Added hypamp stimulus to gid %d: "
                    "delay=%f, duration=%f, amp=%f" %
                    (self.gid, delay, duration, amp), 50)
        return tstim

    def add_replay_relativelinear(self, stimulus):
        """Add a relative linear stimulus."""
        tstim = bglibpy.neuron.h.TStim(0.5, sec=self.soma)
        delay = float(stimulus['Delay'])
        duration = float(stimulus['Duration'])
        amp = (float(stimulus['PercentStart']) / 100.0) * self.threshold
        tstim.pulse(delay, duration, amp)
        self.persistent.append(tstim)

        lazy_printv("Added relative linear stimulus to gid %d: "
                    "delay=%f, duration=%f, amp=%f " %
                    (self.gid, delay, duration, amp), 50)
        return tstim

    def _get_ornstein_uhlenbeck_rand(self, stim_count, seed):
        """Return rng for ornstein_uhlenbeck simulation."""
        if self.rng_settings.mode == "Random123":
            seed1 = stim_count + 2997  # stimulus block
            seed2 = self.rng_settings.stimulus_seed + 291204  # stimulus type
            seed3 = self.gid + 123 if seed is None else seed  # GID
            lazy_printv("Using ornstein_uhlenbeck process seeds %d %d %d" %
                        (seed1, seed2, seed3), 50)
            rng = bglibpy.neuron.h.Random()
            rng.Random123(seed1, seed2, seed3)
        else:
            raise BGLibPyError("Shot noise stimulus requires Random123")

        self.persistent.append(rng)
        return rng

    def _get_shotnoise_step_rand(self, shotnoise_stim_count, seed=None):
        """Return rng for shot noise step stimulus"""
        if self.rng_settings.mode == "Random123":
            seed1 = shotnoise_stim_count + 2997
            seed2 = self.rng_settings.stimulus_seed + 19216
            seed3 = self.gid + 123 if seed is None else seed
            lazy_printv("Using shot noise seeds %d %d %d" %
                        (seed1, seed2, seed3), 50)
            rng = bglibpy.neuron.h.Random()
            rng.Random123(seed1, seed2, seed3)
        else:
            raise BGLibPyError("Shot noise stimulus requires Random123")

        self.persistent.append(rng)
        return rng

    def inject_current_clamp_signal(self, section, segx, tvec, svec):
        """Inject any signal via current clamp."""
        cs = bglibpy.neuron.h.new_IClamp(segx, sec=section)
        cs.dur = tvec[-1]
        svec.play(cs._ref_amp, tvec, 1)

        self.persistent.append(cs)
        self.persistent.append(tvec)
        self.persistent.append(svec)
        return tvec, svec

    def inject_dynamic_clamp_signal(self, section, segx, tvec, svec, reversal):
        """Injects any signal via a dynamic conductance clamp.

        Args:
            reversal (float): reversal potential of conductance (mV)
        """
        clamp = bglibpy.neuron.h.SEClamp(segx, sec=section)
        clamp.dur1 = tvec[-1]
        clamp.amp1 = reversal
        # support delay with initial zero
        tvec.insrt(0, 0)
        svec.insrt(0, 0)
        # replace svec with inverted and clamped signal
        # rs is in MOhm, so conductance is in uS (micro Siemens)
        svec = bglibpy.neuron.h.Vector(
            [1 / x if x > 1E-9 and x < 1E9 else 1E9 for x in svec])
        svec.play(clamp._ref_rs, tvec, 1)

        self.persistent.append(clamp)
        self.persistent.append(tvec)
        self.persistent.append(svec)
        return tvec, svec

    def add_replay_shotnoise(
            self,
            section,
            segx,
            stimulus,
            shotnoise_stim_count=None):
        """Add a replay shot noise stimulus."""
        delay = float(stimulus["Delay"])
        duration = float(stimulus["Duration"])

        dt = float(stimulus.get("Dt", 0.25))

        tau_R = float(stimulus["RiseTime"])
        tau_D = float(stimulus["DecayTime"])
        if tau_R >= tau_D:
            raise BGLibPyError("Shot noise bi-exponential rise time "
                               "must be smaller than decay time")

        rate = float(stimulus["Rate"])
        amp_mean = float(stimulus["AmpMean"])
        amp_var = float(stimulus["AmpVar"])

        seed = int(stimulus["Seed"]) if "Seed" in stimulus else None

        lazy_printv("Adding shot noise stimulus to gid %d: "
                    "delay=%f, duration=%f, rate=%f, amp_mean=%f, "
                    "amp_var=%f tau_D=%f tau_R=%f" %
                    (self.gid, delay, duration, rate, amp_mean,
                     amp_var, tau_D, tau_R), 50)

        rng = self._get_shotnoise_step_rand(shotnoise_stim_count, seed)
        tvec, svec = gen_shotnoise_signal(tau_D, tau_R, rate, amp_mean,
                                          amp_var, duration, dt, rng=rng)
        tvec.add(delay)  # add delay

        if stimulus["Mode"] == "Conductance":
            reversal = float(stimulus.get("Reversal", 0.0))
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_replay_relative_shotnoise(
            self,
            section,
            segx,
            stimulus,
            shotnoise_stim_count=0):
        """Add a replay relative shot noise stimulus."""
        delay = float(stimulus["Delay"])
        duration = float(stimulus["Duration"])

        dt = float(stimulus.get("Dt", 0.25))

        tau_R = float(stimulus["RiseTime"])
        tau_D = float(stimulus["DecayTime"])
        if tau_R >= tau_D:
            raise BGLibPyError("Shot noise bi-exponential rise time "
                               "must be smaller than decay time")

        mean_perc = float(stimulus["MeanPercent"])
        sd_perc = float(stimulus["SDPercent"])
        amp_cv = float(stimulus["AmpCV"])
        cv_square = amp_cv**2

        stim_mode = stimulus["Mode"]
        rel_prop = self.relativity_proportion(stim_mode)

        mean = mean_perc / 100 * rel_prop
        sd = sd_perc / 100 * rel_prop
        var = sd * sd

        rate, amp_mean, amp_var = get_relative_shotnoise_params(
            mean, var, tau_D, tau_R, cv_square)

        seed = int(stimulus["Seed"]) if "Seed" in stimulus else None

        lazy_printv("Adding relative shot noise stimulus to gid %d: "
                    "delay=%f, duration=%f, mean=%f, var=%f, "
                    "amp_cv=%f tau_D=%f tau_R=%f" %
                    (self.gid, delay, duration, mean, var,
                     amp_cv, tau_D, tau_R), 50)

        rng = self._get_shotnoise_step_rand(shotnoise_stim_count, seed)
        tvec, svec = gen_shotnoise_signal(tau_D, tau_R, rate, amp_mean,
                                          amp_var, duration, dt, rng=rng)
        tvec.add(delay)  # add delay

        if stim_mode == "Conductance":
            reversal = float(stimulus.get("Reversal", 0.0))
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_ornstein_uhlenbeck(self, section, segx, stimulus, stim_count=0):
        """Add an Ornstein-Uhlenbeck process, injected as current or conductance."""
        delay = float(stimulus["Delay"])
        duration = float(stimulus["Duration"])

        tau = float(stimulus["Tau"])  # relaxation time [ms]
        dt = float(stimulus.get("Dt", 0.25))
        sigma = float(stimulus["Sigma"])  # signal stdev [uS]
        if sigma <= 0:
            raise ConfigError("Sigma of stimulus must be > 0.")

        mean = float(stimulus["Mean"])    # signal mean [uS]
        if mean < 0 and abs(mean) > 2 * sigma:
            lazy_printv("warning, ornstein_uhlenbeck signal is mostly zero.", 2)

        seed = int(stimulus["Seed"]) if "Seed" in stimulus else None
        rng = self._get_ornstein_uhlenbeck_rand(stim_count, seed)
        tvec, svec = gen_ornstein_uhlenbeck(tau, sigma, mean, duration, dt, rng)

        tvec.add(delay)  # add delay

        if stimulus["Mode"] == "Conductance":
            reversal = float(stimulus.get("Reversal", 0.0))
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_relative_ornstein_uhlenbeck(self, section, segx, stimulus, stim_count=0):
        """Add an Ornstein-Uhlenbeck process, injected as current or conductance,
        relative to cell threshold current or inverse input resistance."""
        delay = float(stimulus["Delay"])
        duration = float(stimulus["Duration"])

        tau = float(stimulus["Tau"])  # relaxation time [ms]
        dt = float(stimulus.get("Dt", 0.25))

        mean_perc = float(stimulus["MeanPercent"])
        sigma_perc = float(stimulus["SDPercent"])

        stim_mode = stimulus["Mode"]
        rel_prop = self.relativity_proportion(stim_mode)

        sigma = sigma_perc / 100 * rel_prop
        if sigma <= 0:
            raise BGLibPyError(f"standard deviation: {sigma}, must be positive.")

        mean = mean_perc / 100 * rel_prop
        if mean < 0 and abs(mean) > 2 * sigma:
            warnings.warn(f"relative ornstein uhlenbeck signal is mostly zero.")

        seed = int(stimulus["Seed"]) if "Seed" in stimulus else None
        rng = self._get_ornstein_uhlenbeck_rand(stim_count, seed)
        tvec, svec = gen_ornstein_uhlenbeck(tau, sigma, mean, duration, dt, rng)

        tvec.add(delay)  # add delay

        if stim_mode == "Conductance":
            reversal = float(stimulus.get("Reversal", 0.0))
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def inject_current_waveform(self, t_content, i_content, section=None,
                                segx=0.5):
        """Inject a custom current to the cell."""
        start_time = t_content[0]
        stop_time = t_content[-1]
        time = bglibpy.neuron.h.Vector()
        currents = bglibpy.neuron.h.Vector()
        time = time.from_python(t_content)
        currents = currents.from_python(i_content)

        if section is None:
            section = self.soma
        pulse = bglibpy.neuron.h.new_IClamp(segx, sec=section)
        self.persistent.append(pulse)
        self.persistent.append(time)
        self.persistent.append(currents)
        setattr(pulse, 'del', start_time)
        pulse.dur = stop_time - start_time
        currents.play(pulse._ref_amp, time)
        return currents

    @tools.deprecated("inject_current_waveform")
    def injectCurrentWaveform(self, t_content, i_content, section=None,
                              segx=0.5):
        """Inject a current in the cell."""
        return self.inject_current_waveform(t_content, i_content, section, segx)

    @tools.deprecated("add_sin_current")
    def addSineCurrentInject(self, start_time, stop_time, freq,
                             amplitude, mid_level, dt=1.0):
        """Add a sinusoidal current injection.

        Returns
        -------

        (numpy array, numpy array) : time and current data

        """
        t_content = np.arange(start_time, stop_time, dt)
        i_content = [amplitude * math.sin(freq * (x - start_time) * (
            2 * math.pi)) + mid_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)
        return (t_content, i_content)

    def add_sin_current(self, amp, start_time, duration, frequency,
                        section=None, segx=0.5):
        """Add a sinusoidal current to the cell."""
        if section is None:
            section = self.soma
        tstim = bglibpy.neuron.h.TStim(segx, sec=section)
        tstim.sin(amp, start_time, duration, frequency)
        self.persistent.append(tstim)
        return tstim
