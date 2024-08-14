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
"""Unit tests for circuit/validate.py."""
import pandas as pd
import pytest

from bluecellulab.circuit.synapse_properties import SynapseProperty
from bluecellulab.circuit.validate import check_nrrp_value


def test_check_nrrp_value():
    """Unit test for check nrrp value."""
    synapses = pd.DataFrame(data={SynapseProperty.NRRP: [15.0, 16.0]})

    check_nrrp_value(synapses)

    synapses[SynapseProperty.NRRP].loc[0] = 15.1
    with pytest.raises(ValueError):
        check_nrrp_value(synapses)

    synapses[SynapseProperty.NRRP].loc[0] = -1

    with pytest.raises(ValueError):
        check_nrrp_value(synapses)
