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
"""Distance computing functionality between Neuron sections."""

import neuron
import numpy as np


class EuclideanSectionDistance:
    """Calculate euclidian distance between positions on two sections.

    Parameters
    ----------

    hsection1 : hoc section such as cADpyr_L2TPC_bluecellulab[0].apic[1]
                First section
    hsection2 : hoc section
                Second section
    location1 : float
                range x along hsection1
    location2 : float
                range x along hsection2
    dimensions : string
                 planes to project on, e.g. 'xy'
    """

    def __call__(
            self,
            hsection1,
            hsection2,
            location1: float = 0.5,
            location2: float = 0.5,
            dimensions: str = "xyz",
    ):
        """Computes and returns the distance."""
        xs_interp1, ys_interp1, zs_interp1 = self.grindaway(hsection1)
        xs_interp2, ys_interp2, zs_interp2 = self.grindaway(hsection2)

        x1 = xs_interp1[int(np.floor((len(xs_interp1) - 1) * location1))]
        y1 = ys_interp1[int(np.floor((len(ys_interp1) - 1) * location1))]
        z1 = zs_interp1[int(np.floor((len(zs_interp1) - 1) * location1))]

        x2 = xs_interp2[int(np.floor((len(xs_interp2) - 1) * location2))]
        y2 = ys_interp2[int(np.floor((len(ys_interp2) - 1) * location2))]
        z2 = zs_interp2[int(np.floor((len(zs_interp2) - 1) * location2))]

        distance = 0
        if "x" in dimensions:
            distance += (x1 - x2) ** 2
        if "y" in dimensions:
            distance += (y1 - y2) ** 2
        if "z" in dimensions:
            distance += (z1 - z2) ** 2

        distance = np.sqrt(distance)

        return distance

    @staticmethod
    def grindaway(hsection):
        """Grindaway."""
        # get the data for the section
        n_segments = int(neuron.h.n3d(sec=hsection))
        n_comps = hsection.nseg

        xs = np.zeros(n_segments)
        ys = np.zeros(n_segments)
        zs = np.zeros(n_segments)
        lengths = np.zeros(n_segments)
        for index in range(n_segments):
            xs[index] = neuron.h.x3d(index, sec=hsection)
            ys[index] = neuron.h.y3d(index, sec=hsection)
            zs[index] = neuron.h.z3d(index, sec=hsection)
            lengths[index] = neuron.h.arc3d(index, sec=hsection)

        # to use Vector class's .interpolate()
        # must first scale the independent variable
        # i.e. normalize length along centroid
        lengths /= lengths[-1]

        # initialize the destination "independent" vector
        # range = np.array(n_comps+2)
        comp_range = np.arange(0, n_comps + 2) / n_comps - 1.0 / (2 * n_comps)
        comp_range[0] = 0
        comp_range[-1] = 1

        # length contains the normalized distances of the pt3d points
        # along the centroid of the section.  These are spaced at
        # irregular intervals.
        # range contains the normalized distances of the nodes along the
        # centroid of the section.  These are spaced at regular intervals.
        # Ready to interpolate.

        xs_interp = np.interp(comp_range, lengths, xs)
        ys_interp = np.interp(comp_range, lengths, ys)
        zs_interp = np.interp(comp_range, lengths, zs)

        return xs_interp, ys_interp, zs_interp
