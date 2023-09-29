"""Unit tests for the RNGSettings class."""

import bluecellulab
from bluecellulab.exceptions import UndefinedRNGException


def test_setting_rngmodes():
    """Test the setting of rng mode."""
    rng_obj = bluecellulab.RNGSettings(mode="Compatibility")
    assert bluecellulab.neuron.h.rngMode == 0

    rng_obj.mode = "Random123"
    assert bluecellulab.neuron.h.rngMode == 1

    rng_obj.mode = "UpdatedMCell"
    assert bluecellulab.neuron.h.rngMode == 2

    bluecellulab.RNGSettings(mode="Random123")
    assert bluecellulab.neuron.h.rngMode == 1
    assert rng_obj.mode == "Random123"

    bluecellulab.RNGSettings()
    assert bluecellulab.neuron.h.rngMode == 1
    assert rng_obj.mode == "Random123"

    try:
        rng_obj.mode = "MersenneTwister"
    except Exception as e:
        assert isinstance(e, UndefinedRNGException)

    # make sure only one object is created
    assert rng_obj is bluecellulab.RNGSettings()


def test_str_repr_obj():
    """Test the str and repr methods of RNGSettings."""
    rng_obj = bluecellulab.RNGSettings(mode="UpdatedMCell")
    assert repr(rng_obj) == "RNGSettings(mode=UpdatedMCell, base_seed=0, " \
                            "synapse_seed=0, " \
                            "ionchannel_seed=0, stimulus_seed=0, " \
                            "minis_seed=0)"

    assert str(rng_obj) == repr(rng_obj)
