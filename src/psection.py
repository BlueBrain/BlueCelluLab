import pylab
import bglibpy
from bglibpy import neuron


class PSection:
    """Class that represents a cell section"""
    def __init__(self, hsection, pparent):
        self.L = hsection.L
        self.diam = hsection.diam
        self.hsection = hsection
        self.name = neuron.h.secname(sec=hsection)
        self.pparent = pparent
        self.hchildren = [neuron.h.SectionRef(sec=self.hsection).child[index] for index in range(0, int(neuron.h.SectionRef(sec=self.hsection).nchild()))]
        self.pchildren = []
        for hchild in self.hchildren:
            self.pchildren.append(PSection(hchild, self))
        self.isLeaf = False
        if not self.hchildren:
            self.isLeaf = True

        self.psegments = []
        self.maxsegdiam = 0
        for hsegment in hsection:
            psegment = bglibpy.PSegment(hsection(hsegment.x), self)
            self.psegments.append(psegment)
            self.maxsegdiam = max(self.maxsegdiam, psegment.diam)

        self.xSpacing = 10
        self.ySpacing = 1

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        """Setup draw of psection"""
        y_accum = 0
        for psegment in self.psegments:
            psegment.setupDraw(figure, x + (self.maxsegdiam - psegment.diam) / 2, y + y_accum, variable=variable, varbounds=varbounds)
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
                varbounds[0] = min(varbounds[0], child_varbounds[0]) if varbounds[0] else child_varbounds[0]
                varbounds[1] = max(varbounds[1], child_varbounds[1]) if varbounds[1] else child_varbounds[1]

        return varbounds

    def drawTree(self, figure, x, y, variable=None, varbounds=None):
        """Draw a dendritic tree"""
        self.setupDraw(figure, x, y, variable=variable, varbounds=varbounds)
        new_x = x  # + self.L + self.xSpacing
        new_y = y + self.L + self.xSpacing
        for child in self.pchildren:
            child.drawTree(figure, new_x, new_y, variable=variable, varbounds=varbounds)
            pylab.plot([x + self.diam / 2, new_x + child.diam / 2], [y + self.L, new_y], 'k')
            new_x = new_x + child.treeWidth()

    def treeWidth(self):
        """Width of a dendritic tree"""
        if self.isLeaf:
            treeWidth = self.maxsegdiam + self.ySpacing
        else:
            treeWidth = 0
            for child in self.pchildren:
                treeWidth += child.treeWidth()

        return max(self.diam + self.ySpacing, treeWidth)

    def treeHeight(self):
        """Height of dendritic tree"""
        return self.L + self.xSpacing + (max([child.treeHeight() for child in self.pchildren]) if self.pchildren else 0)

    def getHChildren(self):
        """All hoc children of a section"""
        return self.hchildren
