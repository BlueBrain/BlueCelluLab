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
"""Unit tests for the exceptions module."""

from bluecellulab.exceptions import error_context


def test_error_context():
    """Unit test for the error_context function."""
    attr_err_msg = "hoc.HocObject' object has no attribute 'minis_single_vesicle_"
    lookup_err_msg = "'X' is not a defined hoc variable name"
    context_info = "mechanism/s for minis_single_vesicle need to be compiled"
    try:
        with error_context(context_info):
            raise AttributeError(attr_err_msg)
    except AttributeError as error:
        assert str(error) == f"{context_info}: {attr_err_msg}"
    try:
        with error_context(context_info):
            raise LookupError(lookup_err_msg)
    except LookupError as error:
        assert str(error) == f"{context_info}: {lookup_err_msg}"
