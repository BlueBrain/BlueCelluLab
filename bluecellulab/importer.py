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

import os
import pkg_resources


def _nrn_disable_banner():
    """Disable NEURON banner."""

    import importlib.util
    import ctypes

    neuron_spec = importlib.util.find_spec("neuron")
    nrnpy_path = neuron_spec.submodule_search_locations[0]
    import glob
    hoc_so_list = \
        glob.glob(os.path.join(nrnpy_path, 'hoc*.so'))

    if len(hoc_so_list) != 1:
        raise Exception(
            'hoc shared library not found in %s' %
            nrnpy_path)

    hoc_so = hoc_so_list[0]
    nrndll = ctypes.cdll[hoc_so]
    ctypes.c_int.in_dll(nrndll, 'nrn_nobanner_').value = 1


def import_neuron():
    """Import NEURON simulator."""
    _nrn_disable_banner()

    import neuron

    return neuron


def import_mod_lib(neuron):
    """Import mod files."""

    mod_lib_list = None
    if 'BGLIBPY_MOD_LIBRARY_PATH' in os.environ:

        # Check if the current directory contains 'x86_64'.

        if os.path.isdir('x86_64'):
            raise Exception("BGLIBPY_MOD_LIBRARY_PATH is set"
                            " and current directory contains the x86_64 folder."
                            " Please remove one of them.")

        mod_lib_path = os.environ["BGLIBPY_MOD_LIBRARY_PATH"]
        mod_lib_list = mod_lib_path.split(':')
        for mod_lib in mod_lib_list:
            neuron.h.nrn_load_dll(mod_lib)

    return mod_lib_list


def import_neurodamus(neuron):
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


def print_header(neuron, mod_lib_path):
    """Print bluecellulab header to stdout."""
    print("Imported neuron from %s" % neuron.__file__)
    print('Mod libs: ', mod_lib_path)


neuron = import_neuron()
mod_lib_paths = import_mod_lib(neuron)
import_neurodamus(neuron)
# print_header(neuron, mod_lib_paths)
