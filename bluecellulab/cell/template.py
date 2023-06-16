# Copyright 2012-2023 Blue Brain Project / EPFL

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

import datetime
import hashlib
import os
import re
import string
from typing import Optional

import bluecellulab
from bluecellulab import lazy_printv, neuron
from bluecellulab.circuit import EmodelProperties
from bluecellulab.exceptions import BluecellulabError


class NeuronTemplate:
    """NeuronTemplate representation."""

    used_template_names: set[str] = set()

    def __init__(
        self, template_filepath: str, morph_filepath: str
    ) -> None:
        """Load the hoc template and init object."""
        self.template_name = self.load(template_filepath)
        self.morph_filepath = morph_filepath

    def get_cell(
        self, template_format: str, gid: int, emodel_properties: Optional[EmodelProperties]
    ) -> neuron.hoc.HocObject:
        """Returns the hoc object matching the template format."""
        morph_dir, morph_fname = os.path.split(self.morph_filepath)
        if template_format == "v6":
            attr_names = getattr(
                neuron.h, self.template_name + "_NeededAttributes", None
            )
            if attr_names is not None:
                if emodel_properties is None:
                    raise BluecellulabError(
                        "EmodelProperties must be provided for template "
                        "format v6 that specifies _NeededAttributes"
                    )
                cell = getattr(neuron.h, self.template_name)(
                    gid,
                    morph_dir,
                    morph_fname,
                    *[emodel_properties.__getattribute__(name) for name in attr_names.split(";")]
                )
            cell = getattr(neuron.h, self.template_name)(
                gid, morph_dir, morph_fname
            )
        elif template_format == "v6_ais_scaler":
            if emodel_properties is None:
                raise BluecellulabError(
                    "EmodelProperties must be provided for template "
                    "format v6_ais_scaler"
                )
            cell = getattr(neuron.h, self.template_name)(
                gid, morph_dir, morph_fname, emodel_properties.ais_scaler
            )
        else:
            cell = getattr(neuron.h, self.template_name)(gid, self.morph_filepath)

        return cell

    @classmethod
    def load(cls, template_filename: str) -> str:
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

        neuron_versiondate_string = bluecellulab.neuron.h.nrnversion(4)
        neuron_versiondate = datetime.datetime.strptime(
            neuron_versiondate_string, "%Y-%m-%d"
        ).date()
        good_neuron_versiondate = datetime.date(2014, 3, 20)

        if neuron_versiondate >= good_neuron_versiondate:
            lazy_printv(
                "This Neuron version supports renaming " "templates, enabling...", 5
            )
            # add bluecellulab to the template name, so that we don't interfere with
            # templates load outside of bluecellulab
            template_name = "%s_bluecellulab" % template_name
            template_name = get_neuron_compliant_template_name(template_name)
            if template_name in cls.used_template_names:
                new_template_name = template_name
                while new_template_name in cls.used_template_names:
                    new_template_name = "%s_x" % new_template_name
                    new_template_name = get_neuron_compliant_template_name(
                        new_template_name
                    )

                template_name = new_template_name

            cls.used_template_names.add(template_name)
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

            bluecellulab.neuron.h(template_content)
        else:
            lazy_printv(
                "This Neuron version doesn't support renaming "
                "templates, disabling...",
                5,
            )
            bluecellulab.neuron.h.load_file(template_filename)

        return template_name


def shorten_and_hash_string(label, keep_length=40, hash_length=9):
    """Convert string to a shorter string if required.

    Parameters
    ----------
    label : string
            a string to be converted
    keep_length : int
                    length of the original string to keep. Default is 40
                    characters.
    hash_length : int
                    length of the hash to generate, should not be more then
                    20. Default is 9 characters.

    Returns
    -------
    new_label : string
        If the length of the original label is shorter than the sum of
        'keep_length' and 'hash_length' plus one the original string is
        returned. Otherwise, a string with structure <partial>_<hash> is
        returned, where <partial> is the first part of the original string
        with length equal to <keep_length> and the last part is a hash of
        'hash_length' characters, based on the original string.
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


def check_compliance_with_neuron(template_name):
    """Verify that a given name is compliant with the rules for a NEURON.

    Parameters
    ----------
    template name : string
                    a name should be a non-empty alphanumeric string,
                    and start with a letter. Underscores are allowed.
                    The length should not exceed 50 characters.

    Returns
    -------
    compliant : boolean
                True if compliant, false otherwise.
    """
    max_len = 50
    return (
        template_name
        and template_name[0].isalpha()
        and template_name.replace("_", "").isalnum()
        and len(template_name) <= max_len
    )


def get_neuron_compliant_template_name(name):
    """Get template name that is compliant with NEURON based on given name.

    Parameters
    ----------
    name : string
            template_name to transform

    Returns
    -------
    new_name : string
                If `name` is NEURON-compliant, the same string is return.
                Otherwise, hyphens are replaced by underscores and if
                appropriate, the string is shortened.
                Leading numbers are removed.
    """
    template_name = name
    if not check_compliance_with_neuron(template_name):
        template_name = template_name.lstrip(string.digits).replace("-", "_")
        template_name = shorten_and_hash_string(
            template_name, keep_length=40, hash_length=9
        )
        lazy_printv(
            "Converted template name %s to %s to make it "
            "NEURON compliant" % (name, template_name),
            50,
        )
    return template_name