#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class that represents a dendrogram window

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

import numpy

# pylint: disable=R0914,too-many-arguments


class Dendrogram(object):

    """Class that represent a dendrogram plot"""

    def __init__(
            self,
            psections,
            variable=None,
            active=False,
            save_fig_path=None,
            interactive=False,
            scale_bar=True,
            scale_bar_size=10.0,
            fig_title=None,
            fig_show=True):
        import pylab

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

        # self.hroot = neuron.h.SectionRef(sec=self.sections[0]).root
        self.proot = psections[0]
        # self.psections = [self.proot] + self.proot.getAllPDescendants()

        xSpacing = self.proot.xSpacing
        ySpacing = self.proot.ySpacing

        max_y = self.proot.treeHeight() + self.proot.ySpacing + title_space
        max_x = self.proot.treeWidth() + self.proot.xSpacing + scale_bar_size
        pylab.xlim([0, max_x])
        pylab.ylim([0, max_y])
        pylab.gca().set_xticks([])
        pylab.gca().set_yticks([])
        pylab.gcf().subplots_adjust(
            top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.3)

        if variable is "v" or variable is None:
            varbounds = [-100, 50]
        else:
            varbounds = self.proot.getTreeVarBounds(variable)

        if variable is not None:
            cax = pylab.imshow(numpy.outer(numpy.arange(0, 1, 0.1), numpy.ones(
                1)), aspect='auto', cmap=pylab.get_cmap("hot"), origin="lower")
            pylab.clim(varbounds[0], varbounds[1])

            cbar = self.dend_figure.colorbar(
                cax, ticks=[varbounds[0], varbounds[1]])
            cbar.ax.set_yticklabels(["%.2e" % (
                varbounds[0]), "%.2e" % (varbounds[1])])

        self.proot.drawTree(self.dend_figure, self.proot.xSpacing,
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
                ('m', 'apical'), ('k', 'soma'), ('b', 'AIS'), ('r', 'basal')]
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

        for secid in self.psections:
            psections[secid].redraw()

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

    def redraw(self):
        """Redraw the dendrogram"""
        if self.active:
            if not self.drawCount:
                for psection in self.psections:
                    psection.redraw()
                self.canvas.blit(self.ax.bbox)
                self.drawCount = 1
            else:
                self.drawCount = self.drawCount - 1

        return True