"""Unit tests for the synapse_factory module."""


import json
from pathlib import Path
import pandas as pd
import pytest
from bluecellulab.cell.core import Cell

from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit.synapse_properties import SynapseProperties, SynapseProperty, synapse_property_decoder
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.synapse import SynapseFactory
from bluecellulab.synapse.synapse_factory import SynapseType
from bluecellulab.synapse.synapse_types import GluSynapse


parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v6
class TestSynapseFactory:

    def setup(self):
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0)
        self.cell = Cell(
            parent_dir / "examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
            parent_dir / "examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
            template_format="v6",
            emodel_properties=emodel_properties,
        )
        with open("tests/test_synapse/test-synapse-series.json") as f:
            self.syn_description = pd.Series(json.load(f, object_hook=synapse_property_decoder))

    def test_create_synapse(self):
        syn_id = ("a", 0)
        condition_parameters = Conditions.init_empty()
        popids = (0, 0)
        extracellular_calcium = 2.0
        connection_modifiers = {
            'Weight': 1.15143,
            'SpontMinis': 0.0,
            'SynapseConfigure': ['%s.Use = 1 %s.Use_GB = 1 %s.Use_p = 1 %s.gmax0_AMPA = gmax_p_AMPA %s.rho_GB = 1 %s.rho0_GB = 1 %s.gmax_AMPA = %s.gmax_p_AMPA']
        }

        synapse = SynapseFactory.create_synapse(
            self.cell, syn_id, self.syn_description, condition_parameters, popids, extracellular_calcium, connection_modifiers
        )
        assert isinstance(synapse, GluSynapse)
        assert synapse.weight == connection_modifiers["Weight"]

    def test_determine_synapse_location(self):
        res = SynapseFactory.determine_synapse_location(self.syn_description, self.cell)
        assert res.location == 0.9999999
        # set afferent_section_pos
        self.syn_description["afferent_section_pos"] = 1.2
        res = SynapseFactory.determine_synapse_location(self.syn_description, self.cell)
        assert res.location == 1.2
        assert res.section.L == pytest.approx(9.530376893488256)

    def test_synlocation_to_segx(self):
        ipt = 13
        isec = 169
        syn_offset = 1.2331762313842773
        section = self.cell.get_hsection(isec)
        res = SynapseFactory.synlocation_to_segx(section, ipt, syn_offset)
        assert res == pytest.approx(0.9999999)
        res = SynapseFactory.synlocation_to_segx(section, ipt, syn_offset=-1.0)
        assert res == pytest.approx(0.9999999)


def test_determine_synapse_type():
    # Mocking a pd.Series for syn_description
    syn_description = pd.Series({
        SynapseProperty.TYPE: 50,
        SynapseProperties.plasticity[0]: None,
        SynapseProperties.plasticity[1]: float('nan')
    })

    # Test when synapse is inhibitory
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.GABAAB

    # Test when synapse is excitatory with no plasticity
    syn_description[SynapseProperty.TYPE] = 150
    for prop in SynapseProperties.plasticity:
        syn_description[prop] = float('nan')  # All properties are NaN
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.AMPANMDA

    # Test when synapse is excitatory with all plasticity properties filled
    for prop in SynapseProperties.plasticity:
        syn_description[prop] = 1.0  # Assign a valid value to each plasticity property
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.GLUSYNAPSE

    # Test when synapse is excitatory with one plasticity property as NaN (should raise an exception)
    syn_description[SynapseProperties.plasticity[1]] = float('nan')
    with pytest.raises(BluecellulabError, match="SynapseFactory: Cannot determine synapse type"):
        SynapseFactory.determine_synapse_type(syn_description)
