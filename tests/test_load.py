#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Testing the loading of the bluecellulab"""

import pytest


@pytest.mark.unit
def test_moduleload():
    """bluecellulab: Test the loading of the bluecellulab module"""
    import bluecellulab  # NOQA


@pytest.mark.unit
def test_verbose_env():
    """bluecellulab: Test env verbose level var"""
    import os
    os.environ['BLUECELLULAB_VERBOSE_LEVEL'] = '10'

    import bluecellulab  # NOQA

    bluecellulab.set_verbose_from_env()

    assert bluecellulab.ENV_VERBOSE_LEVEL == "10"
    assert bluecellulab.VERBOSE_LEVEL == 10

    del os.environ['BLUECELLULAB_VERBOSE_LEVEL']

    bluecellulab.set_verbose_from_env()
    assert bluecellulab.ENV_VERBOSE_LEVEL is None
    assert bluecellulab.VERBOSE_LEVEL == 10

    bluecellulab.set_verbose(0)
    assert bluecellulab.VERBOSE_LEVEL == 0
