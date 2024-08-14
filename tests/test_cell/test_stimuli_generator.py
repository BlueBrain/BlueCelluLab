# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the stimuli_generator module."""

from pytest import approx

import neuron
from bluecellulab.cell.stimuli_generator import (
    gen_shotnoise_signal,
    get_relative_shotnoise_params,
    gen_ornstein_uhlenbeck,
)


def test_gen_shotnoise_signal():
    """Test if the shotnoise signal is generated correctly."""
    rng = neuron.h.Random()
    rng.Random123(1, 2, 3)
    time_vec, stim_vec = gen_shotnoise_signal(4.0, 0.4, 2E3, 40E-3, 16E-4, 2,
                                              rng=rng)
    assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
    assert list(stim_vec) == approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0700357,
                                     0.1032799, 0.1170881, 0.1207344, 0.0])

    # test with None rng
    time_vec, stim_vec = gen_shotnoise_signal(4.0, 0.4, 2E3, -40E-3, 16E-4, 2,
                                              rng=None)


def test_get_relative_shotnoise_params():
    """Unit test for _get_relative_shotnoise_params."""
    rate, amp_mean, amp_var = get_relative_shotnoise_params(
        mean=40e-3, sd=0.04, tau_D=4.0, tau_R=0.4, relative_skew=0.5
    )
    assert rate == approx(227.27272727272705)
    assert amp_mean == approx(0.03406760203796963)
    assert amp_var == approx(0.0011606015086174707)


def test_gen_ornstein_uhlenbeck():
    """Test if the ornstein uhlenbeck signal is generated correctly."""
    rng = neuron.h.Random()
    rng.Random123(1, 2, 3)
    time_vec, stim_vec = gen_ornstein_uhlenbeck(2.8, 0.0042, 0.029, 2, rng=rng)
    assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                     1.75, 2.0, 2.0])
    assert list(stim_vec) == approx(
        [0.029, 0.02933925, 0.02959980, 0.03052109, 0.02882802, 0.03156533,
            0.03289219, 0.03357043, 0.03049419, 0.0])

    # test with None rng
    time_vec, stim_vec = gen_ornstein_uhlenbeck(1e-10, 0.0042, 0.029, 2, rng=None)
