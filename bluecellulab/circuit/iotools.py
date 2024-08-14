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
"""Input/output operations from circuits and simulations."""

from __future__ import annotations
from pathlib import Path
import logging

import bluepy
import numpy as np

from bluecellulab.circuit.node_id import CellId

logger = logging.getLogger(__name__)


def parse_outdat(path: str | Path) -> dict[CellId, np.ndarray]:
    """Parse the replay spiketrains in a out.dat formatted file pointed to by
    path."""
    spikes = bluepy.impl.spike_report.SpikeReport.load(path).get()
    # convert Series to DataFrame with 2 columns for `groupby` operation
    spike_df = spikes.to_frame().reset_index()
    if (spike_df["t"] < 0).any():
        logger.warning('Found negative spike times in out.dat ! '
                       'Clipping them to 0')
        spike_df["t"].clip(lower=0., inplace=True)

    outdat = spike_df.groupby("gid")["t"].apply(np.array)
    # convert outdat's index from int to CellId
    outdat.index = [CellId("", gid) for gid in outdat.index]
    return outdat.to_dict()
