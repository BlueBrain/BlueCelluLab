"""Unit tests for the synapse module."""

import numpy as np
from pytest import approx

from bluecellulab.synapse import Synapse


def test_calc_u_scale_factor():
    """Test calculation of u_scale_factor against its old implementation."""
    def hill(extracellular_calcium, y, K_half):
        return y * extracellular_calcium**4 / (
            K_half**4 + extracellular_calcium**4)

    def constrained_hill(K_half):
        y_max = (K_half**4 + 16) / 16
        return lambda x: hill(x, y_max, K_half)

    def f_scale(x, y):
        return constrained_hill(x)(y)

    scale_factor = np.vectorize(f_scale)

    a = 4.62799366
    b = 3.27495564

    assert scale_factor(a, 2) == approx(Synapse.calc_u_scale_factor(a, 2))
    assert scale_factor(a, 2.2) == approx(Synapse.calc_u_scale_factor(a, 2.2))
    assert scale_factor(a, b) == approx(Synapse.calc_u_scale_factor(a, b))
