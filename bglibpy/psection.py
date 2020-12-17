#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Represents a python version of Neuron Section (for drawing)

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import bglibpy
from bglibpy import neuron


class PSection(object):

    """Class that represents a cell section"""

    def __init__(self, hsection, isec=None):
        self.L = hsection.L
        self.diam = hsection.diam
        self.hsection = hsection
        self.name = neuron.h.secname(sec=hsection)
        self.href = neuron.h.SectionRef(sec=self.hsection)
        self.pparent = None
        self.pchildren = []
        self.isec = isec

        if 'apic' in self.name:
            self.section_type = 'apical'
        elif 'dend' in self.name:
            self.section_type = 'basal'
        elif 'soma' in self.name:
            self.section_type = 'somatic'
        elif 'axon' in self.name:
            self.section_type = 'axonal'
        else:
            raise Exception(
                "PSection: Section of unknown type: %s" %
                self.name)

        self.psegments = []
        self.maxsegdiam = 0
        for hsegment in hsection:
            # psegment = bglibpy.PSegment(hsection(hsegment.x), self)
            psegment = bglibpy.PSegment(hsegment, self)
            self.psegments.append(psegment)
            self.maxsegdiam = max(self.maxsegdiam, psegment.diam)

        self.xSpacing = 1
        self.ySpacing = 5

    @property
    def isLeaf(self):
        """Return true if section is a leaf in the morphological structure."""
        return not self.hchildren

    @property
    def hparent(self):
        """Return the hoc section of the parent"""
        try:
            hparent = self.href().parent
        except SystemError:
            hparent = None
        return hparent

    @property
    def hchildren(self):
        """Return a list with the hoc sections of the children"""
        return [self.href.child[index] for index in
                range(0, int(self.href.nchild()))]

    def add_pchild(self, pchild):
        """Add a python represent of a child section"""
        self.pchildren.append(pchild)

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        """Setup draw of psection"""
        y_accum = 0
        for psegment in self.psegments:
            psegment.setupDraw(figure,
                               x + (self.maxsegdiam - psegment.diam) / 2,
                               y + y_accum,
                               variable=variable,
                               varbounds=varbounds)
            y_accum += psegment.L

    def redraw(self):
        """Redraw psection"""
        for psegment in self.psegments:
            psegment.redraw()

    def getSectionVarBounds(self, variable):
        """Get bounds a variable in a section"""
        varmin = None
        varmax = None
        for psegment in self.psegments:
            value = psegment.getVariableValue(variable)
            if value:
                varmin = min(value, varmin) if varmin else value
                varmax = max(value, varmax) if varmax else value
        return [varmin, varmax]

    def getTreeVarBounds(self, variable):
        """Get the bounds of a variable in a dendritic subtree"""
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
        """Return all the psection that are descendants of this psection"""
        pdescendants = [x for x in self.pchildren]
        for child in self.pchildren:
            pdescendants += child.getAllPDescendants()
        return pdescendants

    def drawTree(self, figure, x, y, variable=None, varbounds=None):
        """Draw a dendritic tree"""
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
        """Width of a dendritic tree"""
        if self.isLeaf:
            treeWidth = self.maxsegdiam + self.xSpacing
        else:
            treeWidth = 0
            for child in self.pchildren:
                treeWidth += child.treeWidth()

        return max(self.diam + self.xSpacing, treeWidth)

    def treeHeight(self):
        """Height of dendritic tree"""
        return self.L + self.ySpacing + \
            (max([child.treeHeight() for child in self.pchildren])
             if self.pchildren else 0)

    def getHChildren(self):
        """All hoc children of a section"""
        return self.hchildren
