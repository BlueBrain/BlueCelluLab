# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
