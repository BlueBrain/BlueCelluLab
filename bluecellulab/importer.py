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
"""Main importer of bluecellulab."""

import logging
import os
from types import ModuleType
import pkg_resources

import neuron

from bluecellulab.exceptions import BluecellulabError


logger = logging.getLogger(__name__)


def import_mod_lib(neuron: ModuleType) -> str:
    """Import mod files."""
    res = ""
    if 'BLUECELLULAB_MOD_LIBRARY_PATH' in os.environ:
        # Check if the current directory contains 'x86_64'.
        if os.path.isdir('x86_64'):
            raise BluecellulabError("BLUECELLULAB_MOD_LIBRARY_PATH is set"
                                    " and current directory contains the x86_64 folder."
                                    " Please remove one of them.")

        mod_lib_path = os.environ["BLUECELLULAB_MOD_LIBRARY_PATH"]
        if mod_lib_path.endswith(".so"):
            neuron.h.nrn_load_dll(mod_lib_path)
        else:
            neuron.load_mechanisms(mod_lib_path)
        res = mod_lib_path
    elif os.path.isdir('x86_64'):
        # NEURON 8.* automatically load these mechamisms
        res = os.path.abspath('x86_64')
    else:
        res = "No mechanisms are loaded."

    return res


def import_neurodamus(neuron: ModuleType) -> None:
    """Import neurodamus."""
    neuron.h.load_file("stdrun.hoc")  # nrn
    hoc_files = [
        "Cell.hoc",  # ND
        "TDistFunc.hoc",  # ND, test dependency
        "TStim.hoc",  # ND
    ]

    for hoc_file in hoc_files:
        hoc_file_path = pkg_resources.resource_filename("bluecellulab", f"hoc/{hoc_file}")
        neuron.h.load_file(hoc_file_path)


def print_header(neuron: ModuleType, mod_lib_path: str) -> None:
    """Print bluecellulab header to stdout."""
    logger.info(f"Imported NEURON from: {neuron.__file__}")
    logger.info(f"Mod lib: {mod_lib_path}")


mod_lib_paths = import_mod_lib(neuron)
import_neurodamus(neuron)
print_header(neuron, mod_lib_paths)
