# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates stimuli to be injected into cells."""

import math
import logging

import neuron

from bluecellulab.cell.random import gamma

logger = logging.getLogger(__name__)


def gen_shotnoise_signal(tau_D, tau_R, rate, amp_mean, amp_var,
                         duration, dt=0.25, rng=None):
    """Adds a Poisson shot noise signal with gamma-distributed amplitudes and
    bi-exponential impulse response.

    tau_D: bi-exponential decay time [ms]
    tau_R: bi-exponential rise time [ms]
    rate: Poisson event rate [Hz]
    amp_mean: mean of gamma-distributed amplitudes [nA]
    amp_var: variance of gamma-distributed amplitudes [nA^2]
    duration: duration of signal [ms]
    dt: timestep [ms]
    rng: random number generator object
    """
    if rng is None:
        logger.info("Using a default RNG for shot noise generation")
        rng = neuron.h.Random()  # Creates a default RNG

    tvec = neuron.h.Vector()
    tvec.indgen(0, duration, dt)  # time vector
    ntstep = len(tvec)  # total number of timesteps

    rate_ms = rate / 1000  # rate in 1 / ms [mHz]
    napprox = 1 + int(duration * rate_ms)  # approximate number of events, at least one
    napprox = int(napprox + 3 * math.sqrt(napprox))  # better bound, as in elephant

    exp_scale = 1 / rate  # scale parameter of exponential distribution of time intervals
    rng.negexp(exp_scale)
    iei = neuron.h.Vector(napprox)
    iei.setrand(rng)  # generate inter-event intervals

    ev = neuron.h.Vector()
    ev.integral(iei, 1).mul(1000)  # generate events in ms
    # add events if last event falls short of duration
    while ev[-1] < duration:
        iei_new = neuron.h.Vector(100)  # generate 100 new inter-event intervals
        iei_new.setrand(rng)            # here rng is still negexp
        ev_new = neuron.h.Vector()
        ev_new.integral(iei_new, 1).mul(1000).add(ev[-1])  # generate new shifted events in ms
        ev.append(ev_new)  # append new events
    ev.where("<", duration)  # remove events exceeding duration
    ev.div(dt)  # divide events by timestep

    nev = neuron.h.Vector([round(x) for x in ev])  # round to integer timestep index
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

    E = neuron.h.Vector(ntstep, 0)  # full signal
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

    P = neuron.h.Vector(ntstep, 0)
    B = neuron.h.Vector(ntstep, 0)

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


def get_relative_shotnoise_params(mean, var, tau_D, tau_R, cv_square):
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


def gen_ornstein_uhlenbeck(tau, sigma, mean, duration, dt=0.25, rng=None):
    """Adds an Ornstein-Uhlenbeck process with given correlation time, standard
    deviation and mean value.

    tau: correlation time [ms], white noise if zero
    sigma: standard deviation [uS]
    mean: mean value [uS]
    duration: duration of signal [ms]
    dt: timestep [ms]
    rng: random number generator object
    """

    if rng is None:
        logger.info("Using a default RNG for Ornstein-Uhlenbeck process")
        rng = neuron.h.Random()  # Creates a default RNG

    tvec = neuron.h.Vector()
    tvec.indgen(0, duration, dt)  # time vector
    ntstep = len(tvec)  # total number of timesteps

    svec = neuron.h.Vector(ntstep, 0)  # stim vector

    noise = neuron.h.Vector(ntstep)  # Gaussian noise
    rng.normal(0.0, 1.0)
    noise.setrand(rng)  # generate Gaussian noise

    if tau < 1e-9:
        svec = noise.mul(sigma)  # white noise
    else:
        mu = math.exp(-dt / tau)  # auxiliar factor [unitless]
        A = sigma * math.sqrt(1 - mu * mu)  # amplitude [uS]
        noise.mul(A)  # scale noise by amplitude [uS]

        # Exact update formula (independent of dt) from Gillespie 1996
        for n in range(1, ntstep):
            svec.x[n] = svec[n - 1] * mu + noise[n]  # signal [uS]

    svec.add(mean)  # shift signal by mean value [uS]

    # append zero at end
    tvec.append(duration)
    svec.append(.0)

    return tvec, svec
