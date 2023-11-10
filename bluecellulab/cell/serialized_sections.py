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
"""Module that allows morphology sections to be accessed from an array by
index."""

import logging
import warnings
import neuron
from bluecellulab.type_aliases import HocObjectType


logger = logging.getLogger(__name__)
warnings.filterwarnings("once", category=UserWarning, module=__name__)


class SerializedSections:

    def __init__(self, cell: HocObjectType) -> None:
        self.isec2sec = {}
        n = cell.nSecAll

        for index, sec in enumerate(cell.all, start=1):
            v_value = sec(0.0001).v
            if v_value >= n:
                logging.debug(f"{sec.name()} v(1)={sec(1).v} n3d()={sec.n3d()}")
                raise ValueError("Error: failure in mk2_isec2sec()")

            if v_value < 0:
                warnings.warn(
                    f"[Warning] SerializedSections: v(0.0001) < 0. index={index} v()={v_value}")
            else:
                self.isec2sec[int(v_value)] = neuron.h.SectionRef(sec=sec)
