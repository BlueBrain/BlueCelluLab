import bglibpy


# Gamma-distributed sample generator (not available in NEURON)
def gamma(rng, a, b, N=1):
    """
    Sample N variates from a gamma distribution with parameters shape = a, scale = b
    using the NEURON random number generator rng.
    Uses the algorithm by Marsaglia and Tsang 2001.
    """
    from math import log, sqrt

    if a < 1:
        rng.uniform(0, 1)
        w = bglibpy.neuron.h.Vector(N)
        w.setrand(rng)
        w.pow(1 / a)
        return gamma(rng, 1 + a, b, N).mul(w)

    d = a - 1 / 3
    c = 1 / 3 / sqrt(d)

    vec = bglibpy.neuron.h.Vector(N)
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
