"""A simple ball and stick model."""

import pkg_resources
from bluecellulab.cell import Cell


def create_ball_stick() -> Cell:
    """Creates a ball and stick model by using package's hoc and asc."""
    hoc = pkg_resources.resource_filename("bluecellulab", "cell/ballstick/emodel.hoc")
    morphology = pkg_resources.resource_filename("bluecellulab", "cell/ballstick/morphology.asc")
    return Cell(template_path=hoc, morphology_path=morphology)
