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
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.warning('This is a warning')
    log_output = stream.getvalue()
    assert log_output == ''

    # Verify that a warning logger entry is displayed when the logger level is set to warning
    logger.setLevel(logging.WARNING)
    logger.warning('This is a warning')
    log_output = stream.getvalue()
    assert log_output == 'This is a warning\n'

    logger.removeHandler(handler)
