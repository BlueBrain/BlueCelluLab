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
"""Class that represents a plot window in bluecellulab."""

import itertools


class PlotWindow:
    """Class the represents a plotting window."""

    def __init__(self, var_list, cell, xlim, ylim, title):
        import pylab
        self.cell = cell
        self.var_list = var_list
        pylab.ion()
        self.figure = pylab.figure(figsize=(10, 10))
        # pylab.ioff()

        self.ax = self.figure.gca()
        self.canvas = self.ax.figure.canvas

        self.figure.suptitle(title)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel("ms")
        self.ax.set_ylabel("mV")

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.line = {}

        linenumber = 0
        for var_name in self.var_list:
            recording = self.cell.get_recording(var_name)
            if recording:
                time = self.cell.get_time()
            else:
                time = self.cell.get_time()[1:]

            # print dir(pylab.gca()._get_lines)
            # print pylab.gca()._get_lines.color_cycle
            # Sorry, don't see a way but disable this warning to access this
            colors = pylab.rcParams['axes.prop_cycle'].by_key()['color']

            linecolors = list(itertools.islice(itertools.cycle(colors), 0, 50))

            self.line[var_name] = pylab.Line2D(
                time, recording, label=var_name,
                color=linecolors[linenumber % len(linecolors)])
            self.ax.add_line(self.line[var_name])
            linenumber += 1

        self.ax.legend()

        self.figure.canvas.draw()

        self.drawCount = 10

    def redraw(self):
        """Redraw the plot window."""
        if not self.drawCount:
            time = self.cell.get_time()
            for var_name in self.var_list:
                voltage = self.cell.get_recording(var_name)
                self.line[var_name].set_data(time, voltage)
                self.ax.draw_artist(self.line[var_name])
            self.canvas.blit(self.ax.bbox)
            self.drawCount = 10
        else:
            self.drawCount = self.drawCount - 1
