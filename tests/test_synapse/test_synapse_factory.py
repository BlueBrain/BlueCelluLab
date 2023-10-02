"""Unit tests for the synapse_factory module."""


import json
from pathlib import Path
import pandas as pd
import pytest
from bluecellulab.cell.core import Cell

from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit.synapse_properties import synapse_property_decoder
from bluecellulab.synapse import SynapseFactory
from bluecellulab.synapse.synapse_factory import SynapseType
from bluecellulab.synapse.synapse_types import GluSynapse


parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v6
def test_create_synapse():
    """Unit test for create_synapse."""
    emodel_properties = EmodelProperties(threshold_current=1.1433533430099487,
                                         holding_current=1.4146618843078613,
                                         ais_scaler=1.4561502933502197,
                                         soma_scaler=1.0)
    cell = Cell(
        "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
        "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
        template_format="v6_adapted",
        emodel_properties=emodel_properties)
    location = 0.5
    syn_id = ("a", 0)
    with open("tests/test_synapse/test-synapse-series.json") as f:
        syn_description = pd.Series(json.load(f, object_hook=synapse_property_decoder))
    condition_parameters = Conditions.init_empty()
    popids = (0, 0)
    extracellular_calcium = 2.0
    connection_modifiers = {
        'Weight': 1.15143,
        'SpontMinis': 0.0,
        'SynapseConfigure': ['%s.Use = 1 %s.Use_GB = 1 %s.Use_p = 1 %s.gmax0_AMPA = gmax_p_AMPA %s.rho_GB = 1 %s.rho0_GB = 1 %s.gmax_AMPA = %s.gmax_p_AMPA']
    }

    synapse = SynapseFactory.create_synapse(
        cell, location, syn_id, syn_description, condition_parameters, popids, extracellular_calcium, connection_modifiers
    )
    assert isinstance(synapse, GluSynapse)
    assert synapse.weight == connection_modifiers["Weight"]


def test_determine_synapse_type():
    """Unit test for determine_synapse_type."""
    assert SynapseFactory.determine_synapse_type(True, False) == SynapseType.GABAAB
    assert SynapseFactory.determine_synapse_type(True, True) == SynapseType.GABAAB

    # Test when synapse is excitatory with plasticity
    assert SynapseFactory.determine_synapse_type(False, True) == SynapseType.GLUSYNAPSE

    # Test when synapse is excitatory without plasticity
    assert SynapseFactory.determine_synapse_type(False, False) == SynapseType.AMPANMDA
