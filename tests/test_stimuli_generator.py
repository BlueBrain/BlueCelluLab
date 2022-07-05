"""Unit tests for the stimuli_generator module."""

from pytest import approx

import bglibpy
from bglibpy.cell.stimuli_generator import gen_shotnoise_signal, get_relative_shotnoise_params


def test_gen_shotnoise_signal():
    """Test if the shotnoise signal is generated correctly."""
    rng = bglibpy.neuron.h.Random()
    rng.Random123(1, 2, 3)
    time_vec, stim_vec = gen_shotnoise_signal(4.0, 0.4, 2E3, 40E-3, 16E-4, 2,
                                                            rng=rng)
    assert list(time_vec) == approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
    assert list(stim_vec) == approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0700357,
                                        0.1032799, 0.1170881, 0.1207344, 0.0])

def test_get_relative_shotnoise_params():
    """Unit test for _get_relative_shotnoise_params."""
    rate, amp_mean, amp_var = get_relative_shotnoise_params(
        mean=40e-3, var=16e-4, tau_D=4.0, tau_R=0.4, cv_square=0.63**2
    )
    assert rate == approx(158.73863636363635)
    assert amp_mean == approx(0.048776006926722876)
    assert amp_var == approx(0.0009442643342459686)
