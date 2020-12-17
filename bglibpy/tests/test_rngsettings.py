"""Unit tests for the RNGSettings class."""

import bglibpy


def test_setting_rngmodes():
    """Test the setting of rng mode."""
    rng_obj = bglibpy.RNGSettings(mode="Compatibility")
    assert(bglibpy.neuron.h.rngMode == 0)

    rng_obj.mode = "Random123"
    assert(bglibpy.neuron.h.rngMode == 1)

    rng_obj.mode = "UpdatedMCell"
    assert(bglibpy.neuron.h.rngMode == 2)

    try:
        rng_obj.mode = "MersenneTwister"
    except Exception as e:
        assert isinstance(e, bglibpy.UndefinedRNGException)

    # make sure only one object is created
    assert (rng_obj is bglibpy.RNGSettings())
