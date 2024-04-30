"""Graph representation of Cells and Synapses."""

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


from bluecellulab.cell.cell_dict import CellDict


def build_graph(cells: CellDict) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add nodes (cells) to the graph
    for cell_id, cell in cells.items():
        G.add_node(cell_id)

    # Extract and add edges (connections) to the graph from each cell
    for cell_id, cell in cells.items():
        for connection in cell.connections.values():
            # Check if pre_cell exists for the connection
            if connection.pre_cell is None:
                continue

            # Source is the pre_cell from the connection
            source_cell_id = connection.pre_cell.cell_id

            # Target is the post-synapse cell from the connection
            target_cell_id = connection.post_synapse.post_cell_id

            # Check if both source and target cells are within the current cell collection
            if source_cell_id in cells and target_cell_id in cells:
                G.add_edge(source_cell_id, target_cell_id, weight=connection.weight)

    return G


def plot_graph(G: nx.Graph, node_size: float = 400, edge_width: float = 0.4, node_distance: float = 1.6):
    # Extract unique populations from the graph nodes
    populations = list(set([cell_id.population_name for cell_id in G.nodes()]))

    # Create a color map for each population
    color_map = plt.cm.tab20(np.linspace(0, 1, len(populations)))  # type: ignore[attr-defined]
    population_color = dict(zip(populations, color_map))

    # Create node colors based on their population
    node_colors = [population_color[node.population_name] for node in G.nodes()]

    # Extract weights for edge color mapping
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    edge_colors = plt.cm.Greens(np.interp(edge_weights, (min(edge_weights), max(edge_weights)), (0.3, 1)))  # type: ignore[attr-defined]

    # Create positions using spring layout for the entire graph
    pos = nx.spring_layout(G, k=node_distance)

    # Create labels only for the node ID
    labels = {node: node.id for node in G.nodes()}

    # Create a figure and axis for the drawing
    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
            edge_color=edge_colors, width=edge_width, node_size=node_size, ax=ax, connectionstyle='arc3, rad = 0.1')

    # Draw directed edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width, ax=ax, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3, rad = 0.1')

    # Create a legend
    for population, color in population_color.items():
        plt.plot([0], [0], color=color, label=population)
    plt.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.05))  # Adjust these values as needed

    # Add a colorbar for edge weights
    sm = ScalarMappable(cmap="Greens", norm=Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label('Synaptic Strength')

    # Add text at the bottom of the figure
    plt.figtext(0.5, 0.01, "Network of simulated cells", ha="center", fontsize=10, va="bottom")

    return plt
