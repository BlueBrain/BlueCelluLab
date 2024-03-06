"""A simple ball and stick model."""

import importlib_resources as resources
from bluecellulab.cell import Cell


def create_ball_stick() -> Cell:
    """Creates a ball and stick model by using package's hoc and asc."""
    hoc = resources.files("bluecellulab") / "cell/ballstick/emodel.hoc"
    morphology = resources.files("bluecellulab") / "cell/ballstick/morphology.asc"
    return Cell(template_path=hoc, morphology_path=morphology)
