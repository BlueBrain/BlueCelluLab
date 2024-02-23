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
"""Contains probability distributions."""

from math import log, sqrt

import neuron


from bluecellulab.type_aliases import NeuronRNG, NeuronVector


# Gamma-distributed sample generator (not available in NEURON)
def gamma(rng: NeuronRNG, a: float, b: float, N: int = 1) -> NeuronVector:
    """Sample N variates from a gamma distribution with parameters shape = a,
    scale = b using the NEURON random number generator rng.

    Uses the algorithm by Marsaglia and Tsang 2001.
    """
    if a < 1:
        rng.uniform(0, 1)
        w = neuron.h.Vector(N)
        w.setrand(rng)
        w.pow(1 / a)
        return gamma(rng, 1 + a, b, N).mul(w)

    d = a - 1 / 3
    c = 1 / 3 / sqrt(d)

    vec = neuron.h.Vector(N)
    for i in range(0, N):
        while True:
            x = rng.normal(0, 1)
            v = 1 + c * x
            if v > 0:
                v = v * v * v
                u = rng.uniform(0, 1)
                if u < 1 - 0.0331 * x * x * x * x:
                    vec.x[i] = b * d * v
                    break
                if log(u) < 0.5 * x * x + d * (1 - v + log(v)):
                    vec.x[i] = b * d * v
                    break

    return vec
