"""Unit tests for cell/random module."""
from pytest import approx

from bluecellulab import neuron
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
