import bglibpy
import matplotlib as plt


class PSegment:
    """A python representation of a segment"""
    def __init__(self, hsegment, parentsection):
        self.hsegment = hsegment
        self.parentsection = parentsection
        self.L = self.parentsection.L / self.parentsection.hsection.nseg
        self.diam = self.hsegment.diam

        self.figure = None
        self.figX = None
        self.figY = None
        self.color_map = plt.cm.get_cmap("hot")
        self.ax = None
        self.patch = None
        self.plotvariable = None
        self.mechanism_names = [mechanism.name() for mechanism in hsegment]
        self.varbounds = None


def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        """Set up the drawing of a segment"""
        self.figure = figure
        self.plotvariable = variable
        self.varbounds = varbounds
        self.ax = self.figure.gca()
        self.figX = x
        self.figY = y
        self.patch = plt.patches.Rectangle([self.figX, self.figY], self.diam, self.L, facecolor="white", edgecolor="black")
        self.ax.add_patch(self.patch)


def redraw(self):
    """Redraw a segment"""
    if self.plotvariable:
        plotvariable_value = self.getVariableValue(self.plotvariable)
        if not plotvariable_value is None:
            self.patch.set_facecolor(self.color_map((plotvariable_value - self.varbounds[0]) / (self.varbounds[1] - self.varbounds[0])))
        else:
            self.patch.set_facecolor(self.color_map(1.0))
            self.patch.set_hatch("/")
        self.ax.draw_artist(self.patch)


def getVariableValue(self, variable):
        """Get a variable value in a segment"""
        if variable is "v" or bglibpy.neuron.h.execute1("{%s.%s(%f)}" % (bglibpy.neuron.h.secname(sec=self.parentsection.hsection), variable, self.hsegment.x), 0):
            return eval("self.hsegment." + variable)
        else:
            return None
