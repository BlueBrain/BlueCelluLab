# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing the loading of the bluecellulab"""

import pytest
import logging
import io


@pytest.mark.unit
def test_moduleload():
    """bluecellulab: Test the loading of the bluecellulab module"""
    import bluecellulab  # NOQA


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
