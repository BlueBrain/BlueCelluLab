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
"""Module for handling NEURON hoc templates."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import string
from typing import NamedTuple, Optional

import neuron

from bluecellulab.circuit import EmodelProperties
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.type_aliases import HocObjectType

import logging

logger = logging.getLogger(__name__)


def public_hoc_cell(cell: HocObjectType) -> HocObjectType:
    """Retrieve the hoc cell to access public hoc functions/attributes."""
    if hasattr(cell, "getCell"):
        return cell.getCell()
    elif hasattr(cell, "CellRef"):
        return cell.CellRef
    else:
        raise BluecellulabError("""Public cell properties cannot be accessed
         from the hoc model. Either getCell() or CellRef needs to be provided""")


class TemplateParams(NamedTuple):
    template_filepath: str | Path
    morph_filepath: str | Path
    template_format: str
    emodel_properties: Optional[EmodelProperties]


class NeuronTemplate:
    """NeuronTemplate representation."""

    def __init__(
        self, template_filepath: str | Path, morph_filepath: str | Path,
        template_format: str, emodel_properties: Optional[EmodelProperties]
    ) -> None:
        """Load the hoc template and init object."""
        if isinstance(template_filepath, Path):
            template_filepath = str(template_filepath)
        if isinstance(morph_filepath, Path):
            morph_filepath = str(morph_filepath)

        if not os.path.exists(template_filepath):
            raise FileNotFoundError(f"Couldn't find template file: {template_filepath}")
        if not os.path.exists(morph_filepath):
            raise FileNotFoundError(f"Couldn't find morphology file: {morph_filepath}")

        self.template_name = self.load(template_filepath)
        self.morph_filepath = morph_filepath
        self.template_format = template_format
        self.emodel_properties = emodel_properties

    def get_cell(self, gid: Optional[int]) -> HocObjectType:
        """Returns the hoc object matching the template format."""
        morph_dir, morph_fname = os.path.split(self.morph_filepath)
        if self.template_format == "v6":
            attr_names = getattr(
                neuron.h, self.template_name.split('_bluecellulab')[0] + "_NeededAttributes", None
            )
            if attr_names is not None:
                if self.emodel_properties is None:
                    raise BluecellulabError(
                        "EmodelProperties must be provided for template "
                        "format v6 that specifies _NeededAttributes"
                    )
                cell = getattr(neuron.h, self.template_name)(
                    gid,
                    morph_dir,
                    morph_fname,
                    *[self.emodel_properties.__getattribute__(name) for name in attr_names.split(";")]
                )
            else:
                cell = getattr(neuron.h, self.template_name)(
                    gid,
                    morph_dir,
                    morph_fname,
                )
        elif self.template_format == "bluepyopt":
            cell = getattr(neuron.h, self.template_name)(morph_dir, morph_fname)
        else:
            cell = getattr(neuron.h, self.template_name)(gid, self.morph_filepath)

        return cell

    def load(self, template_filename: str) -> str:
        """Read a cell template. If template name already exists, rename it.

        Args:
            template_filename: path string containing template file.

        Returns:
            resulting template name
        """
        with open(template_filename) as template_file:
            template_content = template_file.read()

        match = re.search(r"begintemplate\s*(\S*)", template_content)
        template_name = match.group(1)  # type:ignore

        logger.debug("This Neuron version supports renaming templates, enabling...")
        # add bluecellulab to the template name, so that we don't interfere with
        # templates load outside of bluecellulab
        template_name = "%s_bluecellulab" % template_name
        template_name = get_neuron_compliant_template_name(template_name)
        obj_address = hex(id(self))
        template_name = f"{template_name}_{obj_address}"

        template_content = re.sub(
            r"begintemplate\s*(\S*)",
            "begintemplate %s" % template_name,
            template_content,
        )
        template_content = re.sub(
            r"endtemplate\s*(\S*)",
            "endtemplate %s" % template_name,
            template_content,
        )

        neuron.h(template_content)

        return template_name


def shorten_and_hash_string(label: str, keep_length=40, hash_length=9) -> str:
    """Converts a string to a shorter string if required.

    Args:
        label: A string to be converted.
        keep_length: Length of the original string to keep.
        hash_length: Length of the hash to generate, should not be more than 20.

    Returns:
        If the length of the original label is shorter than the sum of 'keep_length'
        and 'hash_length' plus one, the original string is returned. Otherwise, a
        string with structure <partial>_<hash> is returned, where <partial> is the
        first part of the original string with length equal to <keep_length> and the
        last part is a hash of 'hash_length' characters, based on the original string.
    """
    if hash_length > 20:
        raise ValueError(
            "Parameter hash_length should not exceed 20, "
            " received: {}".format(hash_length)
        )

    if len(label) <= keep_length + hash_length + 1:
        return label

    hash_string = hashlib.sha1(label.encode("utf-8")).hexdigest()
    return "{}_{}".format(label[0:keep_length], hash_string[0:hash_length])


def check_compliance_with_neuron(template_name: str) -> bool:
    """Verify that a given name is compliant with the rules for a NEURON.

    A name should be a non-empty alphanumeric string, and start with a
    letter. Underscores are allowed. The length should not exceed 50
    characters.
    """
    max_len = 50
    return (
        bool(template_name)
        and template_name[0].isalpha()
        and template_name.replace("_", "").isalnum()
        and len(template_name) <= max_len
    )


def get_neuron_compliant_template_name(name: str) -> str:
    """Get template name that is compliant with NEURON based on given name."""
    template_name = name
    if not check_compliance_with_neuron(template_name):
        template_name = template_name.lstrip(string.digits).replace("-", "_")
        template_name = shorten_and_hash_string(
            template_name, keep_length=40, hash_length=9
        )
        logger.debug("Converted template name %s to %s to make it "
                     "NEURON compliant" % (name, template_name))
    return template_name
