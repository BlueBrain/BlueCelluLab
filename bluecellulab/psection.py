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
"""Represents a python version of NEURON Section (for drawing)."""
from __future__ import annotations
import re
import neuron

import bluecellulab
from bluecellulab.cell.serialized_sections import SerializedSections
from bluecellulab.psegment import PSegment
from bluecellulab.type_aliases import HocObjectType, NeuronSection


def init_psections(
    hoc_cell: HocObjectType,
) -> tuple[dict[int, PSection], dict[str, PSection]]:
    """Initialize the psections list.

    This list contains the Python representation of the psections of
    this morphology.
    """
    psections: dict[int, PSection] = {}
    secname_to_psection: dict[str, PSection] = {}
    for sec in hoc_cell.all:
        secname = neuron.h.secname(sec=sec)
        secname_to_psection[secname] = PSection(sec)

    serial_sections = SerializedSections(hoc_cell)
    for isec, val in serial_sections.isec2sec.items():
        hsection: NeuronSection = val.sec
        if hsection:
            secname = neuron.h.secname(sec=hsection)
            psections[isec] = secname_to_psection[secname]
            psections[isec].isec = isec

    # Set the parents and children of all the psections
    for psec in psections.values():
        hparent = psec.hparent
        if hparent:
            parentname = hparent.name()
            psec.pparent = secname_to_psection[parentname]
        else:
            psec.pparent = None

        for hchild in psec.hchildren:
            childname = hchild.name()
            if "myelin" in childname:
                continue
            pchild = secname_to_psection[childname]
            psec.add_pchild(pchild)

    return psections, secname_to_psection


class PSection:
    """Class that represents a cell section."""

    def __init__(self, hsection: NeuronSection, isec: int | None = None):
        self.L: float = hsection.L
        self.diam: float = hsection.diam
        self.hsection = hsection
        self.name: str = neuron.h.secname(sec=hsection)
        self.href: NeuronSection = neuron.h.SectionRef(sec=self.hsection)
        self.pparent: PSection | None = None
        self.pchildren: list[PSection] = []
        self.isec = isec

        self.psegments: list[PSegment] = []
        self.maxsegdiam = 0.0
        for hsegment in hsection:
            # psegment = bluecellulab.PSegment(hsection(hsegment.x), self)
            psegment = bluecellulab.PSegment(hsegment, self)
            self.psegments.append(psegment)
            self.maxsegdiam = max(self.maxsegdiam, psegment.diam)

        self.xSpacing = 1.0
        self.ySpacing = 5.0

    @property
    def section_type(self) -> str:
        """Return the type of the section."""
        # From Cell[0].soma[0] -> soma
        matches = re.findall(r'\.([^.\[\]]+)\[', self.name)
        if matches:
            return matches[-1]  # Return the last match
        return 'unknown'  # Return 'unknown' if no matches are found

    @property
    def is_leaf(self) -> bool:
        """Return true if section is a leaf in the morphological structure."""
        return not self.hchildren

    @property
    def hparent(self) -> NeuronSection | None:
        """Return the hoc section of the parent."""
        if self.href.has_parent():
            return self.href.parent
        else:
            return None

    @property
    def hchildren(self) -> list[NeuronSection]:
        """Return a list with the hoc sections of the children."""
        return [self.href.child[index] for index in
                range(0, int(self.href.nchild()))]

    def add_pchild(self, pchild: PSection) -> None:
        """Add a python represent of a child section."""
        self.pchildren.append(pchild)
        pchild.pparent = self

    def getSectionVarBounds(self, variable):
        """Get bounds a variable in a section."""
        varmin = None
        varmax = None
        for psegment in self.psegments:
            value = psegment.get_variable_value(variable)
            if value:
                varmin = min(value, varmin) if varmin else value
                varmax = max(value, varmax) if varmax else value
        return [varmin, varmax]

    def getTreeVarBounds(self, variable):
        """Get the bounds of a variable in a dendritic subtree."""
        varbounds = self.getSectionVarBounds(variable)
        for child in self.pchildren:
            child_varbounds = child.getTreeVarBounds(variable)
            if child_varbounds[0] and child_varbounds[1]:
                varbounds[0] = min(
                    varbounds[0],
                    child_varbounds[0]) if varbounds[0] else child_varbounds[0]
                varbounds[1] = max(
                    varbounds[1],
                    child_varbounds[1]) if varbounds[1] else child_varbounds[1]

        return varbounds

    def all_descendants(self) -> list[PSection]:
        """Return all the psection that are descendants of this psection."""
        pdescendants = list(self.pchildren)
        for child in self.pchildren:
            pdescendants += child.all_descendants()
        return pdescendants

    def tree_width(self) -> float:
        """Width of a dendritic tree."""
        if self.is_leaf:
            width = self.maxsegdiam + self.xSpacing
        else:
            width = sum(child.tree_width() for child in self.pchildren)
        return max(self.diam + self.xSpacing, width)

    def tree_height(self) -> float:
        """Height of dendritic tree."""
        return self.L + self.ySpacing + \
            (max([child.tree_height() for child in self.pchildren])
             if self.pchildren else 0)
