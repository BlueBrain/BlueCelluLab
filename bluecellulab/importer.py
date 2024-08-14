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
"""Main importer of bluecellulab."""

import importlib_resources as resources
import logging
import os
from types import ModuleType

import neuron

from bluecellulab.exceptions import BluecellulabError
from bluecellulab.utils import CaptureOutput, run_once


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


def import_hoc(neuron: ModuleType) -> None:
    """Import hoc dependencies."""
    neuron.h.load_file("stdrun.hoc")  # nrn
    hoc_files = [
        "Cell.hoc",  # ND
        "TDistFunc.hoc",  # ND, test dependency
        "TStim.hoc",  # ND
    ]

    for hoc_file in hoc_files:
        hoc_file_path = str(resources.files("bluecellulab") / f"hoc/{hoc_file}")
        with CaptureOutput() as stdoud:
            neuron.h.load_file(hoc_file_path)
            logger.debug(f"Loaded {hoc_file}. stdout from the hoc: {stdoud}")


def print_header(neuron: ModuleType, mod_lib_path: str) -> None:
    """Print bluecellulab header to stdout."""
    logger.debug(f"Imported NEURON from: {neuron.__file__}")
    logger.debug(f"Mod lib: {mod_lib_path}")


@run_once
def _load_mod_files() -> None:
    """Import hoc and mod files."""
    logger.debug("Loading the mod files.")
    mod_lib_paths = import_mod_lib(neuron)
    print_header(neuron, mod_lib_paths)


def load_mod_files(func):
    def wrapper(*args, **kwargs):
        _load_mod_files()
        return func(*args, **kwargs)
    return wrapper
