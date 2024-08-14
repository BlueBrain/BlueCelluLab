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
"""Unit tests for cell/random module."""
import neuron
from pytest import approx

from bluecellulab.cell.random import gamma


def test_gamma():
    """Unit test for the gamma function."""
    rng = neuron.h.Random()
    gamma_shape = 0.5
    gamma_scale = 1.5
    N = 5
    res = gamma(rng, gamma_shape, gamma_scale, N)
    assert len(res) == N
    assert sum(res) == approx(2.9341513)
    assert res[0] == approx(0.2862183)
    assert max(res) == approx(1.3015527)
    assert min(res) == approx(0.2802995)
