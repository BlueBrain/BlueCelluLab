from pathlib import Path

import matplotlib.pyplot as plt

from bluecellulab.ssim import SSim
from bluecellulab.graph import build_graph, plot_graph





sonata_sim_path = (
    Path("tests/examples/sim_quick_scx_sonata_multicircuit")
    / f"simulation_config_shotnoise.json"
)

cell_ids = [("NodeA", 0), ("NodeA", 1), ("NodeA", 2),
            ("NodeB", 0), ("NodeB", 1)
            ]

sim = SSim(sonata_sim_path)
sim.instantiate_gids(cell_ids, add_synapses=True)
G = build_graph(sim.cells)

plot_graph(G, node_size=500, edge_width=0.6, node_distance=1.8)
plt.savefig("graph.png", dpi=300)
