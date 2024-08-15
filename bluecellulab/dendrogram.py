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
"""Class that represents a dendrogram window."""
from __future__ import annotations
import numpy as np
import pylab

from bluecellulab.psection import PSection
from bluecellulab.psegment import PSegment


def setup_draw(psegments: list[PSegment], maxsegdiam: float, figure, x, y, variable=None, varbounds=None) -> None:
    """Setup draw of psection."""
    y_accum = 0.0
    for psegment in psegments:
        psegment.setupDraw(figure,
                           x + (maxsegdiam - psegment.diam) / 2,
                           y + y_accum,
                           variable=variable,
                           varbounds=varbounds)
        y_accum += psegment.L


def draw_tree(psection: PSection, figure, x, y, variable=None, varbounds=None) -> None:
    """Draw a dendritic tree."""
    # Draw myself
    setup_draw(
        psection.psegments, psection.maxsegdiam, figure, x, y, variable=variable, varbounds=varbounds
    )

    # Draw children

    # First child is a same x coordinate
    new_x = x  # + self.L + self.xSpacing

    # Children drawn L + ySpacing heigher
    new_y = y + psection.L + psection.ySpacing

    for child in psection.pchildren:
        draw_tree(child, figure, new_x, new_y, variable=variable, varbounds=varbounds)
        pylab.plot(
            [x + psection.diam / 2, new_x + child.diam / 2],
            [y + psection.L, new_y], 'k')
        # Prepare new_x for next child
        new_x = new_x + child.tree_width()


def redraw_psection(psection: PSection) -> None:
    """Redraw psection."""
    for psegment in psection.psegments:
        psegment.redraw()


class Dendrogram:
    """Class that represent a dendrogram plot."""

    def __init__(
            self,
            psections: list[PSection],
            variable=None,
            active=False,
            save_fig_path=None,
            interactive=False,
            scale_bar=True,
            scale_bar_size=10.0,
            fig_title=None,
            fig_show=True):

        if interactive:
            pylab.ion()

        self.dend_figure = pylab.figure(figsize=(20, 12))

        title_space = 0.0
        if fig_title:
            title_space = 30.0
            self.dend_figure.suptitle(fig_title)

        if interactive:
            pylab.ioff()

        self.psections = psections
        # neuron.h.finitialize()

        self.proot: PSection = psections[0]
        self.psections = [self.proot] + self.proot.all_descendants()

        xSpacing = self.proot.xSpacing
        ySpacing = self.proot.ySpacing

        max_y = self.proot.tree_height() + self.proot.ySpacing + title_space
        max_x = self.proot.tree_width() + self.proot.xSpacing + scale_bar_size
        pylab.xlim([0, max_x])
        pylab.ylim([0, max_y])
        pylab.gca().set_xticks([])
        pylab.gca().set_yticks([])
        pylab.gcf().subplots_adjust(
            top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.3)

        if variable == "v" or variable is None:
            varbounds = [-100, 50]
        else:
            varbounds = self.proot.getTreeVarBounds(variable)

        if variable is not None:
            cax = pylab.imshow(np.outer(np.arange(0, 1, 0.1), np.ones(
                1)), aspect='auto', cmap=pylab.get_cmap("hot"), origin="lower")
            pylab.clim(varbounds[0], varbounds[1])

            cbar = self.dend_figure.colorbar(
                cax, ticks=[varbounds[0], varbounds[1]])
            cbar.ax.set_yticklabels(["%.2e" % (
                varbounds[0]), "%.2e" % (varbounds[1])])

        draw_tree(self.proot, self.dend_figure, self.proot.xSpacing,
                  self.proot.ySpacing, variable=variable,
                  varbounds=varbounds)

        if scale_bar:
            pylab.plot(
                [max_x - xSpacing, max_x - xSpacing, max_x - xSpacing, max_x -
                 xSpacing - scale_bar_size],
                [ySpacing, ySpacing + scale_bar_size, ySpacing, ySpacing], 'k',
                linewidth=2)
            pylab.text(
                max_x -
                xSpacing -
                0.9 * scale_bar_size,
                2 *
                ySpacing,
                "10 micron",
                fontsize=8)

            color_types = [
                ('m', 'apical'),
                ('k', 'soma'),
                ('b', 'AIS'),
                ('r', 'basal'),
                ('g', 'myelin')
            ]
            for i, (color, section_type) in enumerate(color_types):
                pylab.text(
                    max_x -
                    xSpacing -
                    0.9 *
                    scale_bar_size,
                    (i + 2) * 3 * scale_bar_size,
                    section_type,
                    fontsize=8,
                    bbox=dict(
                        facecolor=color, alpha=0.5))

        self.dend_figure.canvas.draw()

        for section in self.psections:
            section_id = section.isec
            if section_id is not None:
                redraw_psection(psections[section_id])

        self.canvas = self.dend_figure.gca().figure.canvas
        self.ax = self.dend_figure.gca()
        self.canvas.blit(self.ax.bbox)

        self.background = self.canvas.copy_from_bbox(
            self.dend_figure.gca().bbox)
        self.drawCount = 1

        self.active = active

        if save_fig_path is not None:
            pylab.savefig(save_fig_path)

        if not interactive and fig_show:
            pylab.show()

    def redraw(self) -> None:
        """Redraw the dendrogram."""
        if self.active:
            if not self.drawCount:
                for psection in self.psections:
                    redraw_psection(psection)
                self.canvas.blit(self.ax.bbox)
                self.drawCount = 1
            else:
                self.drawCount = self.drawCount - 1
