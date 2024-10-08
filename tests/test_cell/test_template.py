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
"""Unit tests for template.py module."""

from pathlib import Path
from unittest.mock import Mock, patch
import uuid

import pytest

from bluecellulab.cell.template import NeuronTemplate, public_hoc_cell
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.exceptions import BluecellulabError


examples_dir = Path(__file__).resolve().parent.parent / "examples"

hipp_hoc_path = (
    examples_dir
    / "hippocampus_opt_cell_template"
    / "electrophysiology"
    / "cell.hoc"
)
hipp_morph_path = (
    examples_dir / "hippocampus_opt_cell_template" / "morphology" / "cell.asc"
)

v6_hoc_path = (
    examples_dir / "circuit_sonata_quick_scx" / "components" / "hoc" / "cADpyr_L2TPC.hoc"
)

v6_morph_path = (
    examples_dir / "circuit_sonata_quick_scx" / "components" / "morphologies" / "asc" / "rr110330_C3_idA.asc"
)


@patch('uuid.uuid4')
def test_get_cell_with_bluepyopt_template(mock_uuid):
    """Unit test for the get_cell method with bluepyopt_template."""

    id = '12345678123456781234567812345678'
    mock_uuid.return_value = uuid.UUID(id)
    template = NeuronTemplate(hipp_hoc_path, hipp_morph_path, "bluepyopt", None)
    cell = template.get_cell(gid=None)

    assert cell.hname() == f"bACnoljp_bluecellulab_{id}[0]"


def test_neuron_template_init():
    """Unit test for the NeuronTemplate's constructor."""
    missing_file = "missing_file"

    with pytest.raises(FileNotFoundError):
        NeuronTemplate(missing_file, hipp_morph_path, "bluepyopt", None)
    with pytest.raises(FileNotFoundError):
        NeuronTemplate(hipp_hoc_path, missing_file, "bluepyopt", None)

    NeuronTemplate(hipp_hoc_path, hipp_morph_path, "bluepyopt", None)


def test_public_hoc_cell_bluepyopt_template():
    """Unit test for public_hoc_cell."""
    template = NeuronTemplate(hipp_hoc_path, hipp_morph_path, "bluepyopt", None)
    cell = template.get_cell(None)
    hoc_public = public_hoc_cell(cell)
    assert hoc_public.gid == 0.0


def test_public_hoc_cell_v6_template():
    """Unit test for public_hoc_cell."""
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    template = NeuronTemplate(v6_hoc_path, v6_morph_path, "v6", emodel_properties)
    cell = template.get_cell(5)
    hoc_public = public_hoc_cell(cell)
    assert hoc_public.gid == 5.0


def test_public_hoc_cell_v6_template_raises_bluecellulaberror():
    """Test when NeuronTemplate constructor raises a BluecellulabError."""
    with pytest.raises(BluecellulabError) as excinfo:
        template = NeuronTemplate(v6_hoc_path, v6_morph_path, "v6", emodel_properties=None)
        cell = template.get_cell(5)
    assert "EmodelProperties must be provided for template format v6 that specifies _NeededAttributes" in str(excinfo.value)


def test_public_hoc_cell_failure():
    """Unit test for public_hoc_cell when neither getCell nor CellRef is provided."""
    cell_without_getCell_or_CellRef = Mock(spec=[])  # spec=[] ensures no attributes exist
    with pytest.raises(BluecellulabError) as excinfo:
        public_hoc_cell(cell_without_getCell_or_CellRef)
    assert "Public cell properties cannot be accessed" in str(excinfo.value)


def test_load_bpo_template():
    """Test the loading of a hoc without getCell or gid."""
    hoc_path = examples_dir / "bpo_cell" / "0_cADpyr_L5TPC_a6e707a_1_sNone.hoc"
    morph_path = examples_dir / "bpo_cell" / "C060114A5.asc"
    neuron_template = NeuronTemplate(hoc_path, morph_path, "bluepyopt", None)
    cell = neuron_template.get_cell(None)
    assert len(cell.soma[0].children()) == 11
