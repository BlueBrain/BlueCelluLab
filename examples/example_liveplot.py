"Test detect_hypamp_network"

import bglibpy

blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"
gids = [107462]

bglibpy.VERBOSE_LEVEL = 1

ssim = bglibpy.SSim(blueconfig)

ssim.instantiate_gids(gids, synapse_detail=2, add_replay=True, add_stimuli=True)

cell = ssim.cells[gids[0]]

cell.add_plot_window(["self.soma(0.5)._ref_v"])
cell.add_dendrogram(variable="v", active=True)
ssim.run(1000)
