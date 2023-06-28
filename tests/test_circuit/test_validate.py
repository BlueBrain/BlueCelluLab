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
