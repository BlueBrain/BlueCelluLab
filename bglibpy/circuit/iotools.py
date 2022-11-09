"""Input/output operations from circuits and simulations."""

import bluepy
import numpy as np

from bglibpy import lazy_printv


def parse_outdat(path: str) -> dict:
    """Parse the replay spiketrains in a out.dat formatted file
       pointed to by path"""

    spikes = bluepy.impl.spike_report.SpikeReport.load(path).get()
    # convert Series to DataFrame with 2 columns for `groupby` operation
    spike_df = spikes.to_frame().reset_index()
    if (spike_df["t"] < 0).any():
        lazy_printv('WARNING: SSim: Found negative spike times in out.dat ! '
                    'Clipping them to 0', 2)
        spike_df["t"].clip(lower=0., inplace=True)

    outdat = spike_df.groupby("gid")["t"].apply(np.array)
    return outdat.to_dict()
