#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Testing the loading of the bluecellulab"""

import pytest
import logging
import io


@pytest.mark.unit
def test_moduleload():
    """bluecellulab: Test the loading of the bluecellulab module"""
    import bluecellulab  # NOQA


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
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == 10

    bluecellulab.set_verbose(0)
    assert bluecellulab.VERBOSE_LEVEL == 0
    # Since 0 is equivalent to NOTSET, it will inherit the log level from its parent logger or the root logger.
    assert logging.getLogger('bluecellulab').getEffectiveLevel() == logging.getLogger('bluecellulab').parent.getEffectiveLevel()


@pytest.mark.unit
def test_logger():
    """Test logging"""

    import bluecellulab  # NOQA

    # Verify the hierarchy and level inheritance
    parent_logger = logging.getLogger('bluecellulab')
    parent_logger.setLevel(logging.DEBUG)
    child_logger = logging.getLogger('bluecellulab.' + __name__)
    assert child_logger.getEffectiveLevel() == parent_logger.getEffectiveLevel()
    assert child_logger.parent is parent_logger

    # Verify that a warning logger entry is not displayed when the logger level is set to critical
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.CRITICAL)
    stream = io.StringIO()
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.warning('This is a warning')
    log_output = stream.getvalue()
    assert log_output == ''
