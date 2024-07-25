"""Circuit related helper functions."""

import copy
from pathlib import Path
from bluepy_configfile.configfile import BlueConfig


def blueconfig_append_path(blueconfig, path, fields=None):
    """Appends path to the certain path fields in a given blueconfig.

    Args:
        blueconfig : config object or BlueConfig file path
        fields (list): collection of fields (str) to be modified
        path (str or pathlib.Path): path to be appended to the fields
         of blueconfig

    Returns:
        bluepy_configfile.configfile.BlueConfigFile: modified config object
    """

    # bluepyconfigfile doesn't support pathlib yet
    if isinstance(blueconfig, Path):
        blueconfig = str(blueconfig)
    with open(blueconfig) as f:
        blueconfig = BlueConfig(f)

    if not fields:
        fields = [
            "MorphologyPath",
            "METypePath",
            "CircuitPath",
            "nrnPath",
            "CurrentDir",
            "OutputRoot",
            "TargetFile",
        ]

    new_bc = copy.deepcopy(blueconfig)
    for field in fields:
        old_value = new_bc.Run.__getattr__(field)
        new_value = (Path(path) / old_value).absolute()
        new_bc.Run.__setattr__(field, str(new_value))
    return new_bc
