"""Test for the verbosity module."""


import logging
import pytest


@pytest.mark.unit
def test_verbose_env():
    """bluecellulab: Test env verbose level var"""
    import os
    os.environ['BLUECELLULAB_VERBOSE_LEVEL'] = "10"

    import bluecellulab  # NOQA

    bluecellulab.set_verbose_from_env()

    assert bluecellulab.ENV_VERBOSE_LEVEL == "10"
    assert bluecellulab.VERBOSE_LEVEL == 10

    del os.environ['BLUECELLULAB_VERBOSE_LEVEL']

    bluecellulab.set_verbose_from_env()
    assert bluecellulab.ENV_VERBOSE_LEVEL is None
    assert bluecellulab.VERBOSE_LEVEL == 10
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.DEBUG

    bluecellulab.set_verbose(0)
    assert bluecellulab.VERBOSE_LEVEL == 0
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.CRITICAL

    bluecellulab.set_verbose(1)
    assert bluecellulab.VERBOSE_LEVEL == 1
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.ERROR

    bluecellulab.set_verbose(2)
    assert bluecellulab.VERBOSE_LEVEL == 2
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.WARNING

    bluecellulab.set_verbose(5)
    assert bluecellulab.VERBOSE_LEVEL == 5
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.INFO
