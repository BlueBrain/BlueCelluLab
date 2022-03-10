"""Contains injection functionality for the cell."""

import math

import numpy as np

import bglibpy
from bglibpy import printv, BGLibPyError, tools


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
