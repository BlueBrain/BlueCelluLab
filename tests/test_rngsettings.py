"""Unit tests for the RNGSettings class."""

import neuron

from bluecellulab import RNGSettings
from bluecellulab.exceptions import UndefinedRNGException


def test_setting_rngmodes():
    """Test the setting of rng mode."""
    rng_obj = RNGSettings.get_instance()
    rng_obj.mode = "Compatibility"
    initial_obj_id = id(rng_obj)  # this should never change - Singleton object
    assert neuron.h.rngMode == 0

    rng_obj.mode = "Random123"
    assert neuron.h.rngMode == 1

    rng_obj.mode = "UpdatedMCell"
    assert neuron.h.rngMode == 2

    rng_obj.mode = "Random123"
    assert neuron.h.rngMode == 1
    assert rng_obj.mode == "Random123"
    assert id(rng_obj) == initial_obj_id

    try:
        rng_obj.mode = "MersenneTwister"
    except Exception as e:
        assert isinstance(e, UndefinedRNGException)

    # make sure only one object is created
    assert rng_obj is RNGSettings.get_instance()
    assert id(rng_obj) == id(RNGSettings.get_instance()) == initial_obj_id


def test_str_repr_obj():
    """Test the str and repr methods of RNGSettings."""
    rng_obj = RNGSettings.get_instance()
    rng_obj.set_seeds()
    rng_obj.mode = "UpdatedMCell"
    assert repr(rng_obj) == "RNGSettings(mode=UpdatedMCell, base_seed=0, " \
                            "synapse_seed=0, " \
                            "ionchannel_seed=0, stimulus_seed=0, " \
                            "minis_seed=0)"

    assert str(rng_obj) == repr(rng_obj)
