# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Represent a NEURON Segment in Python (for drawing)."""

import neuron

from bluecellulab.neuron_interpreter import eval_neuron

type_colormap = {'apic': 'm', 'dend': 'r', 'soma': 'k', 'axon': 'b', 'myelin': 'g', 'unknown': 'c'}


class PSegment:
    """A python representation of a segment."""

    def __init__(self, hsegment, parentsection):
        # import matplotlib as plt
        from matplotlib import colormaps

        self.hsegment = hsegment
        self.parentsection = parentsection
        self.L = self.parentsection.L / self.parentsection.hsection.nseg
        self.diam = self.hsegment.diam

        self.figure = None
        self.figX = None
        self.figY = None
        self.color_map = colormaps["hot"]
        self.ax = None
        self.patch = None
        self.plotvariable = None
        self.mechanism_names = [mechanism.name() for mechanism in hsegment]
        self.varbounds = None

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        """Set up the drawing of a segment."""
        import matplotlib as plt

        self.figure = figure
        self.plotvariable = variable
        self.varbounds = varbounds
        self.ax = self.figure.gca()
        self.figX = x
        self.figY = y
        self.patch = plt.patches.Rectangle(
            [self.figX, self.figY], self.diam, self.L,
            facecolor=type_colormap[self.parentsection.section_type],
            edgecolor=type_colormap[self.parentsection.section_type])
        self.ax.add_patch(self.patch)

    def redraw(self):
        """Redraw a segment."""
        if self.plotvariable:
            plotvariable_value = self.get_variable_value(self.plotvariable)
            if plotvariable_value is not None:
                self.patch.set_facecolor(self.color_map(
                    (plotvariable_value - self.varbounds[0]) /
                    (self.varbounds[1] - self.varbounds[0])))
            else:
                self.patch.set_facecolor(self.color_map(1.0))
                self.patch.set_hatch("/")
            self.ax.draw_artist(self.patch)

    def get_variable_value(self, variable):
        """Get a variable value in a segment."""
        if variable == "v" or neuron.h.execute1(
            "{%s.%s(%f)}"
            % (
                neuron.h.secname(sec=self.parentsection.hsection),
                variable,
                self.hsegment.x,
            ),
            0,
        ):
            return eval_neuron(f"self.hsegment.{variable}", self=self)
        else:
            return None
