"""Contains injection functionality for the cell."""

import math

import numpy as np

import bglibpy
from bglibpy import printv, BGLibPyError, tools
from bglibpy.cell.random import gamma


class InjectableMixin:
    """Mixin responsible of injections to the cell.
       Important Usage Note: Adds the instantiated Neuron objects to
        self.persistent to explicitly destroy them when their lifetime ends.
    """

    def add_pulse(self, stimulus):
        """Inject pulse stimulus for replay."""
        tstim = bglibpy.neuron.h.TStim(0.5, sec=self.soma)
        if 'Offset' in stimulus.keys():
            # The meaning of "Offset" is not clear yet, ask Jim
            # delay = float(stimulus.Delay) +
            #        float(stimulus.Offset)
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

        printv("Added noise stimulus to gid %d: "
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
        printv("Added hypamp stimulus to gid %d: "
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

        printv("Added relative linear stimulus to gid %d: "
               "delay=%f, duration=%f, amp=%f " %
               (self.gid, delay, duration, amp), 50)
        return tstim

    def _get_shotnoise_step_rand(self, shotnoise_stim_count, seed=None):
        """Return rng for shot noise step stimulus"""
        if self.rng_settings.mode == "Random123":
            seed1 = shotnoise_stim_count + 2997
            seed2 = self.rng_settings.stimulus_seed + 19216
            seed3 = self.gid + 123 if seed is None else seed
            printv("Using shot noise seeds %d %d %d" %
                   (seed1, seed2, seed3), 50)
            rng = bglibpy.neuron.h.Random()
            rng.Random123(seed1, seed2, seed3)
        else:
            raise BGLibPyError("Shot noise stimulus requires Random123")

        self.persistent.append(rng)
        return rng

    def _gen_shotnoise_signal(self,
                              tau_D, tau_R, rate, amp_mean, amp_var,
                              duration, dt=0.25, rng=None):
        if rng is None:
            printv("Using a default RNG for shot noise generation", 50)
            rng = bglibpy.neuron.h.Random()  # Creates a default RNG

        tvec = bglibpy.neuron.h.Vector()
        tvec.indgen(0, duration, dt)  # time vector
        ntstep = len(tvec)  # total number of timesteps

        rate_ms = rate / 1000  # rate in 1 / ms [mHz]
        napprox = 1 + int(duration * rate_ms)            # approximate number of events, at least one
        napprox = int(napprox + 3 * math.sqrt(napprox))  # better bound, as in elephant

        exp_scale = 1 / rate  # scale parameter of exponential distribution of time intervals
        rng.negexp(exp_scale)
        iei = bglibpy.neuron.h.Vector(napprox)
        iei.setrand(rng)  # generate inter-event intervals

        ev = bglibpy.neuron.h.Vector()
        ev.integral(iei, 1).mul(1000)  # generate events in ms
        # add events if last event falls short of duration
        while ev[-1] < duration:
            iei_new = bglibpy.neuron.h.Vector(100)  # generate 100 new inter-event intervals
            iei_new.setrand(rng)            # here rng is still negexp
            ev_new = bglibpy.neuron.h.Vector()
            ev_new.integral(iei_new, 1).mul(1000).add(ev[-1])  # generate new shifted events in ms
            ev.append(ev_new)  # append new events
        ev.where("<", duration)  # remove events exceeding duration
        ev.div(dt)  # divide events by timestep

        nev = bglibpy.neuron.h.Vector([round(x) for x in ev])  # round to integer timestep index
        nev.where("<", ntstep)  # remove events exceeding number of timesteps

        sign = 1
        # if amplitude mean is negative, invert sign of current
        if amp_mean < 0:
            amp_mean = -amp_mean
            sign = -1

        gamma_scale = amp_var / amp_mean      # scale parameter of gamma distribution
        gamma_shape = amp_mean / gamma_scale  # shape parameter of gamma distribution
        # sample gamma-distributed amplitudes
        amp = gamma(rng, gamma_shape, gamma_scale, len(nev))

        E = bglibpy.neuron.h.Vector(ntstep, 0)  # full signal
        for n, A in zip(nev, amp):
            E.x[int(n)] += sign * A  # add impulses, may overlap due to rounding to timestep

        # perform equivalent of convolution with bi-exponential impulse response
        # through a composite autoregressive process with impulse train as innovations

        # unitless quantities (time measured in timesteps)
        a = math.exp(-dt / tau_D)
        b = math.exp(-dt / tau_R)
        D = -math.log(a)
        R = -math.log(b)
        t_peak = math.log(R / D) / (R - D)
        A = (a / b - 1) / (a ** t_peak - b ** t_peak)

        P = bglibpy.neuron.h.Vector(ntstep, 0)
        B = bglibpy.neuron.h.Vector(ntstep, 0)

        # composite autoregressive process with exact solution
        # P[n] = b * (a ^ n - b ^ n) / (a - b)
        # for unit response B[0] = P[0] = 0, E[0] = 1
        for n in range(1, ntstep):
            P.x[n] = a * P[n - 1] + b * B[n - 1]
            B.x[n] = b * B[n - 1] + E[n - 1]

        P.mul(A)  # normalize to peak amplitude

        # append zero at end
        tvec.append(duration)
        P.append(.0)

        return tvec, P

    def add_shotnoise_step(self, section, segx,
                           tau_D, tau_R, rate, amp_mean, amp_var,
                           delay, duration, dt=0.25,
                           seed=None, shotnoise_stim_count=0):
        """Inject a shot noise current signal."""
        rng = self._get_shotnoise_step_rand(shotnoise_stim_count, seed)
        tvec, svec = self._gen_shotnoise_signal(tau_D, tau_R, rate, amp_mean,
                                                amp_var, duration, dt, rng=rng)
        tvec.add(delay)  # add delay

        cs = bglibpy.neuron.h.new_IClamp(segx, sec=section)
        self.persistent.append(cs)
        self.persistent.append(tvec)
        self.persistent.append(svec)
        cs.dur = tvec[-1]
        svec.play(cs._ref_amp, tvec, 1)

        return tvec, svec

    def _get_relative_shotnoise_params(self, mean, var, tau_D, tau_R, cv_square):
        """Returns Rate, amp_mean and amp_var parameters."""
        # bi-exponential time to peak [ms]
        t_peak = math.log(tau_D / tau_R) / (1 / tau_R - 1 / tau_D)
        # bi-exponential peak height [1]
        x_peak = math.exp(-t_peak / tau_D) - math.exp(-t_peak / tau_R)

        rate_ms = (1 + cv_square) / 2 * (mean ** 2 / var) / (tau_D + tau_R)
        rate = rate_ms * 1000  # rate in 1 / s [Hz]
        amp_mean = mean * x_peak / rate_ms / (tau_D - tau_R)
        amp_var = cv_square * amp_mean ** 2

        return rate, amp_mean, amp_var

    def add_replay_shotnoise(
            self,
            stimulus,
            shotnoise_seed=None,
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

        seed = stimulus.get("Seed")
        if seed is not None:
            seed = int(seed)

        if shotnoise_seed is None:
            shotnoise_seed = seed

        printv("Added shot noise stimulus to gid %d: "
               "delay=%f, duration=%f, rate=%f, amp_mean=%f, "
               "amp_var=%f tau_D=%f tau_R=%f" %
               (self.gid, delay, duration, rate, amp_mean,
                amp_var, tau_D, tau_R), 50)

        return self.add_shotnoise_step(self.soma, 0.5,
                                       tau_D, tau_R, rate, amp_mean, amp_var,
                                       delay, duration, dt, shotnoise_seed,
                                       shotnoise_stim_count)

    def add_replay_relative_shotnoise(
            self,
            stimulus,
            shotnoise_seed=None,
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

        mean_perc = float(stimulus["MeanPercent"])
        sd_perc = float(stimulus["SDPercent"])
        amp_cv = float(stimulus["AmpCV"])
        cv_square = amp_cv * amp_cv

        mean = mean_perc / 100 * self.threshold
        sd = sd_perc / 100 * self.threshold
        var = sd * sd

        rate, amp_mean, amp_var = self._get_relative_shotnoise_params(
            mean, var, tau_D, tau_R, cv_square)

        seed = stimulus.get("Seed")
        if seed is not None:
            seed = int(seed)

        if shotnoise_seed is None:
            shotnoise_seed = seed

        printv("Added relative shot noise stimulus to gid %d: "
               "delay=%f, duration=%f, mean=%f, var=%f, "
               "amp_cv=%f tau_D=%f tau_R=%f" %
               (self.gid, delay, duration, mean, var,
                amp_cv, tau_D, tau_R), 50)

        return self.add_shotnoise_step(self.soma, 0.5,
                                       tau_D, tau_R, rate, amp_mean, amp_var,
                                       delay, duration, dt, shotnoise_seed,
                                       shotnoise_stim_count)

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
