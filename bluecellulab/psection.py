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
"""Represents a python version of NEURON Section (for drawing)."""
from __future__ import annotations
import neuron

import bluecellulab
from bluecellulab.cell.serialized_sections import SerializedSections
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

        if 'apic' in self.name:
            self.section_type = 'apical'
        elif 'dend' in self.name:
            self.section_type = 'basal'
        elif 'soma' in self.name:
            self.section_type = 'somatic'
        elif 'axon' in self.name:
            self.section_type = 'axonal'
        elif 'myelin' in self.name:
            self.section_type = 'myelin'
        else:
            raise Exception(
                "PSection: Section of unknown type: %s" %
                self.name)

        self.psegments = []
        self.maxsegdiam = 0
        for hsegment in hsection:
            # psegment = bluecellulab.PSegment(hsection(hsegment.x), self)
            psegment = bluecellulab.PSegment(hsegment, self)
            self.psegments.append(psegment)
            self.maxsegdiam = max(self.maxsegdiam, psegment.diam)

        self.xSpacing = 1
        self.ySpacing = 5

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

    def add_pchild(self, pchild) -> None:
        """Add a python represent of a child section."""
        self.pchildren.append(pchild)

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        """Setup draw of psection."""
        y_accum = 0
        for psegment in self.psegments:
            psegment.setupDraw(figure,
                               x + (self.maxsegdiam - psegment.diam) / 2,
                               y + y_accum,
                               variable=variable,
                               varbounds=varbounds)
            y_accum += psegment.L

    def redraw(self):
        """Redraw psection."""
        for psegment in self.psegments:
            psegment.redraw()

    def getSectionVarBounds(self, variable):
        """Get bounds a variable in a section."""
        varmin = None
        varmax = None
        for psegment in self.psegments:
            value = psegment.getVariableValue(variable)
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

    def getAllPDescendants(self):
        """Return all the psection that are descendants of this psection."""
        pdescendants = list(self.pchildren)
        for child in self.pchildren:
            pdescendants += child.getAllPDescendants()
        return pdescendants

    def drawTree(self, figure, x, y, variable=None, varbounds=None):
        """Draw a dendritic tree."""
        import pylab

        # Draw myself
        self.setupDraw(figure, x, y, variable=variable, varbounds=varbounds)

        # Draw children

        # First child is a same x coordinate
        new_x = x  # + self.L + self.xSpacing

        # Children drawn L + ySpacing heigher
        new_y = y + self.L + self.ySpacing

        for child in self.pchildren:
            child.drawTree(
                figure, new_x, new_y, variable=variable, varbounds=varbounds)
            pylab.plot(
                [x + self.diam / 2, new_x + child.diam / 2],
                [y + self.L, new_y], 'k')
            # Prepare new_x for next child
            new_x = new_x + child.treeWidth()

    def treeWidth(self):
        """Width of a dendritic tree."""
        if self.is_leaf:
            treeWidth = self.maxsegdiam + self.xSpacing
        else:
            treeWidth = sum(child.treeWidth() for child in self.pchildren)
        return max(self.diam + self.xSpacing, treeWidth)

    def treeHeight(self):
        """Height of dendritic tree."""
        return self.L + self.ySpacing + \
            (max([child.treeHeight() for child in self.pchildren])
             if self.pchildren else 0)
