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
"""Plotting functions for the cell module."""

from __future__ import annotations

import neuron

import bluecellulab


class PlottableMixin:
    """Mixin responsible of plotting functions of a cell."""

    def __init__(self) -> None:
        """Store the persistent objects."""
        self.cell_dendrograms: list[bluecellulab.Dendrogram] = []
        self.plot_windows: list[bluecellulab.PlotWindow] = []

        self.fih_plots = None
        self.fih_weights = None

        # As long as no PlotWindow or active Dendrogram exist, don't update
        self.plot_callback_necessary = False

    def delete_plottable(self) -> None:
        """NEURON state to be cleaned upon object destruction."""
        self.fih_plots = None
        self.fih_weights = None

    def add_plot_window(self, var_list, xlim=None, ylim=None, title=""):
        """Add a window to plot a variable."""
        xlim = [0, 1000] if xlim is None else xlim
        ylim = [-100, 100] if ylim is None else ylim
        for var_name in var_list:
            if var_name not in self.recordings:
                self.add_recording(var_name)
        self.plot_windows.append(bluecellulab.PlotWindow(
            var_list, self, xlim, ylim, title))
        self.plot_callback_necessary = True

    def add_dendrogram(
            self,
            variable=None,
            active=False,
            save_fig_path=None,
            interactive=False,
            scale_bar=True,
            scale_bar_size=10.0,
            fig_title=None):
        """Show a dendrogram of the cell."""
        self._init_psections()
        cell_dendrogram = bluecellulab.Dendrogram(
            self.psections,
            variable=variable,
            active=active,
            save_fig_path=save_fig_path,
            interactive=interactive,
            scale_bar=scale_bar,
            scale_bar_size=scale_bar_size,
            fig_title=fig_title)
        cell_dendrogram.redraw()
        self.cell_dendrograms.append(cell_dendrogram)
        if active:
            self.plot_callback_necessary = True

    def init_callbacks(self):
        """Initialize the callback function (if necessary)."""
        if not self.delayed_weights.empty():
            self.fih_weights = neuron.h.FInitializeHandler(
                1, self.weights_callback)

        if self.plot_callback_necessary:
            self.fih_plots = neuron.h.FInitializeHandler(1, self.plot_callback)

    def weights_callback(self):
        """Callback function that updates the delayed weights, when a certain
        delay has been reached."""
        while not self.delayed_weights.empty() and \
                abs(self.delayed_weights.queue[0][0] - neuron.h.t) < \
                neuron.h.dt:
            (_, (sid, weight)) = self.delayed_weights.get()
            if sid in self.connections:
                if self.connections[sid].post_netcon is not None:
                    self.connections[sid].post_netcon.weight[0] = weight

        if not self.delayed_weights.empty():
            neuron.h.cvode.event(
                self.delayed_weights.queue[0][0], self.weights_callback
            )

    def plot_callback(self):
        """Update all the windows."""
        for window in self.plot_windows:
            window.redraw()
        for cell_dendrogram in self.cell_dendrograms:
            cell_dendrogram.redraw()

        neuron.h.cvode.event(
            neuron.h.t + 1, self.plot_callback)
