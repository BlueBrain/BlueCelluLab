# Copyright 2012-2023 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains injection functionality for the cell."""

import math
import warnings
import logging

import numpy as np

import bluecellulab
from bluecellulab import tools
from bluecellulab.cell.stimuli_generator import (
    gen_ornstein_uhlenbeck,
    gen_shotnoise_signal,
    get_relative_shotnoise_params,
)
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.stimuli import (
    ClampMode,
    Hyperpolarizing,
    Noise,
    OrnsteinUhlenbeck,
    ShotNoise,
    RelativeOrnsteinUhlenbeck,
    RelativeShotNoise,
)
from bluecellulab.type_aliases import HocObjectType


logger = logging.getLogger(__name__)


class InjectableMixin:
    """Mixin responsible of injections to the cell.

    Important Usage Note: Adds the instantiated Neuron objects to
     self.persistent to explicitly destroy them when their lifetime ends.
    """

    def relativity_proportion(self, stim_mode: ClampMode) -> float:
        """Relativity proportion used in Relative stimuli e.g. relative
        shotnoise."""
        if stim_mode == ClampMode.CONDUCTANCE:
            input_resistance = self.sonata_proxy.get_input_resistance().iloc[0]  # type: ignore
            return 1.0 / input_resistance
        else:  # Current
            return self.threshold  # type: ignore

    def add_pulse(self, stimulus):
        """Inject pulse stimulus for replay."""
        tstim = bluecellulab.neuron.h.TStim(0.5, sec=self.soma)
        tstim.train(stimulus.delay,
                    stimulus.duration,
                    stimulus.amp_start,
                    stimulus.frequency,
                    stimulus.width)
        self.persistent.append(tstim)
        return tstim

    def add_step(self, start_time, stop_time, level, section=None, segx=0.5):
        """Add a step current injection."""
        if section is None:
            section = self.soma

        tstim = bluecellulab.neuron.h.TStim(segx, sec=section)
        duration = stop_time - start_time
        tstim.pulse(start_time, duration, level)
        self.persistent.append(tstim)
        return tstim

    def add_ramp(self, start_time, stop_time, start_level, stop_level,
                 section=None, segx=0.5):
        """Add a ramp current injection."""
        if section is None:
            section = self.soma

        tstim = bluecellulab.neuron.h.TStim(segx, sec=section)

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
        """Add a voltage clamp.

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
        vclamp = bluecellulab.neuron.h.SEClamp(segx, sec=section)
        self.persistent.append(vclamp)

        vclamp.amp1 = level
        vclamp.dur1 = stop_time

        if rs is not None:
            vclamp.rs = rs

        current = bluecellulab.neuron.h.Vector()
        if current_record_dt is None:
            current.record(vclamp._ref_i)
        else:
            current.record(
                vclamp._ref_i,
                self.get_precise_record_dt(current_record_dt))

        self.recordings[current_record_name] = current
        return vclamp

    def _get_noise_step_rand(self, noisestim_count):
        """Return rng for noise step stimulus."""
        if self.rng_settings.mode == "Compatibility":
            rng = bluecellulab.neuron.h.Random(self.gid + noisestim_count)
        elif self.rng_settings.mode == "UpdatedMCell":
            rng = bluecellulab.neuron.h.Random()
            rng.MCellRan4(
                noisestim_count * 10000 + 100,
                self.rng_settings.base_seed +
                self.rng_settings.stimulus_seed +
                self.gid * 1000)
        elif self.rng_settings.mode == "Random123":
            rng = bluecellulab.neuron.h.Random()
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
                       duration, seed=None, noisestim_count=0):
        """Inject a step current with noise on top."""
        if seed is not None:
            rand = bluecellulab.neuron.h.Random(seed)
        else:
            rand = self._get_noise_step_rand(noisestim_count)

        tstim = bluecellulab.neuron.h.TStim(segx, rand, sec=section)
        tstim.noise(delay, duration, mean, variance)
        self.persistent.append(rand)
        self.persistent.append(tstim)
        return tstim

    def add_replay_noise(
            self,
            stimulus: Noise,
            noise_seed=None,
            noisestim_count=0):
        """Add a replay noise stimulus."""
        mean = (stimulus.mean_percent * self.threshold) / 100.0  # type: ignore
        variance = (stimulus.variance * self.threshold) / 100.0  # type: ignore
        tstim = self.add_noise_step(
            self.soma,  # type: ignore
            0.5,
            mean,
            variance,
            stimulus.delay,
            stimulus.duration,
            seed=noise_seed,
            noisestim_count=noisestim_count)

        return tstim

    def add_replay_hypamp(self, stimulus: Hyperpolarizing):
        """Inject hypamp for the replay."""
        tstim = bluecellulab.neuron.h.TStim(0.5, sec=self.soma)  # type: ignore
        amp = self.hypamp  # type: ignore
        tstim.pulse(stimulus.delay, stimulus.duration, amp)
        self.persistent.append(tstim)  # type: ignore
        return tstim

    def add_replay_relativelinear(self, stimulus):
        """Add a relative linear stimulus."""
        tstim = bluecellulab.neuron.h.TStim(0.5, sec=self.soma)
        amp = stimulus.percent_start / 100.0 * self.threshold
        tstim.pulse(stimulus.delay, stimulus.duration, amp)
        self.persistent.append(tstim)

        return tstim

    def _get_ornstein_uhlenbeck_rand(self, stim_count, seed):
        """Return rng for ornstein_uhlenbeck simulation."""
        if self.rng_settings.mode == "Random123":
            seed1 = stim_count + 2997  # stimulus block
            seed2 = self.rng_settings.stimulus_seed + 291204  # stimulus type
            seed3 = self.gid + 123 if seed is None else seed  # GID
            logger.debug("Using ornstein_uhlenbeck process seeds %d %d %d" %
                         (seed1, seed2, seed3))
            rng = bluecellulab.neuron.h.Random()
            rng.Random123(seed1, seed2, seed3)
        else:
            raise BluecellulabError("Shot noise stimulus requires Random123")

        self.persistent.append(rng)
        return rng

    def _get_shotnoise_step_rand(self, shotnoise_stim_count, seed=None):
        """Return rng for shot noise step stimulus."""
        if self.rng_settings.mode == "Random123":
            seed1 = shotnoise_stim_count + 2997
            seed2 = self.rng_settings.stimulus_seed + 19216
            seed3 = self.gid + 123 if seed is None else seed
            logger.debug("Using shot noise seeds %d %d %d" %
                         (seed1, seed2, seed3))
            rng = bluecellulab.neuron.h.Random()
            rng.Random123(seed1, seed2, seed3)
        else:
            raise BluecellulabError("Shot noise stimulus requires Random123")

        self.persistent.append(rng)
        return rng

    def inject_current_clamp_signal(self, section, segx, tvec, svec):
        """Inject any signal via current clamp."""
        cs = bluecellulab.neuron.h.IClamp(segx, sec=section)
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
        clamp = bluecellulab.neuron.h.SEClamp(segx, sec=section)
        clamp.dur1 = tvec[-1]
        clamp.amp1 = reversal
        # support delay with initial zero
        tvec.insrt(0, 0)
        svec.insrt(0, 0)
        # replace svec with inverted and clamped signal
        # rs is in MOhm, so conductance is in uS (micro Siemens)
        svec = bluecellulab.neuron.h.Vector(
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
            stimulus: ShotNoise,
            shotnoise_stim_count=0):
        """Add a replay shot noise stimulus."""
        rng = self._get_shotnoise_step_rand(shotnoise_stim_count, stimulus.seed)
        tvec, svec = gen_shotnoise_signal(stimulus.decay_time, stimulus.rise_time, stimulus.rate, stimulus.amp_mean,
                                          stimulus.amp_var, stimulus.duration, stimulus.dt, rng=rng)
        tvec.add(stimulus.delay)  # add delay

        if stimulus.mode == ClampMode.CONDUCTANCE:
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, stimulus.reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_replay_relative_shotnoise(
            self,
            section,
            segx,
            stimulus: RelativeShotNoise,
            shotnoise_stim_count=0):
        """Add a replay relative shot noise stimulus."""
        cv_square = stimulus.amp_cv**2

        stim_mode = stimulus.mode
        rel_prop = self.relativity_proportion(stim_mode)

        mean = stimulus.mean_percent / 100 * rel_prop
        sd = stimulus.sd_percent / 100 * rel_prop
        var = sd * sd

        rate, amp_mean, amp_var = get_relative_shotnoise_params(
            mean, var, stimulus.decay_time, stimulus.rise_time, cv_square)

        rng = self._get_shotnoise_step_rand(shotnoise_stim_count, stimulus.seed)
        tvec, svec = gen_shotnoise_signal(stimulus.decay_time, stimulus.rise_time, rate, amp_mean,
                                          amp_var, stimulus.duration, stimulus.dt, rng=rng)
        tvec.add(stimulus.delay)  # add delay

        if stim_mode == ClampMode.CONDUCTANCE:
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, stimulus.reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_ornstein_uhlenbeck(
        self, section, segx, stimulus: OrnsteinUhlenbeck, stim_count=0
    ):
        """Add an Ornstein-Uhlenbeck process, injected as current or
        conductance."""
        rng = self._get_ornstein_uhlenbeck_rand(stim_count, stimulus.seed)
        tvec, svec = gen_ornstein_uhlenbeck(
            stimulus.tau,
            stimulus.sigma,
            stimulus.mean,
            stimulus.duration,
            stimulus.dt,
            rng,
        )

        tvec.add(stimulus.delay)  # add delay

        if stimulus.mode == ClampMode.CONDUCTANCE:
            return self.inject_dynamic_clamp_signal(
                section, segx, tvec, svec, stimulus.reversal
            )
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def add_relative_ornstein_uhlenbeck(
        self, section, segx, stimulus: RelativeOrnsteinUhlenbeck, stim_count=0
    ):
        """Add an Ornstein-Uhlenbeck process, injected as current or
        conductance, relative to cell threshold current or inverse input
        resistance."""
        stim_mode = stimulus.mode
        rel_prop = self.relativity_proportion(stim_mode)

        sigma = stimulus.sd_percent / 100 * rel_prop
        if sigma <= 0:
            raise BluecellulabError(f"standard deviation: {sigma}, must be positive.")

        mean = stimulus.mean_percent / 100 * rel_prop
        if mean < 0 and abs(mean) > 2 * sigma:
            warnings.warn("relative ornstein uhlenbeck signal is mostly zero.")

        rng = self._get_ornstein_uhlenbeck_rand(stim_count, stimulus.seed)
        tvec, svec = gen_ornstein_uhlenbeck(
            stimulus.tau, sigma, mean, stimulus.duration, stimulus.dt, rng
        )

        tvec.add(stimulus.delay)  # add delay

        if stim_mode == ClampMode.CONDUCTANCE:
            return self.inject_dynamic_clamp_signal(section, segx, tvec, svec, stimulus.reversal)
        else:
            return self.inject_current_clamp_signal(section, segx, tvec, svec)

    def inject_current_waveform(self, t_content, i_content, section=None,
                                segx=0.5):
        """Inject a custom current to the cell."""
        start_time = t_content[0]
        stop_time = t_content[-1]
        time = bluecellulab.neuron.h.Vector()
        currents = bluecellulab.neuron.h.Vector()
        time = time.from_python(t_content)
        currents = currents.from_python(i_content)

        if section is None:
            section = self.soma
        pulse = bluecellulab.neuron.h.IClamp(segx, sec=section)
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
        tstim = bluecellulab.neuron.h.TStim(segx, sec=section)
        tstim.sin(amp, start_time, duration, frequency)
        self.persistent.append(tstim)
        return tstim

    def add_alpha_synapse(
        self,
        onset: float,
        tau: float,
        gmax: float,
        e: float,
        section: HocObjectType,
        segx=0.5,
    ) -> HocObjectType:
        """Add an AlphaSynapse NEURON point process stimulus to the cell."""
        syn = bluecellulab.neuron.h.AlphaSynapse(segx, sec=section)
        syn.onset = onset
        syn.tau = tau
        syn.gmax = gmax
        syn.e = e
        self.persistent.append(syn)  # type: ignore
        return syn
