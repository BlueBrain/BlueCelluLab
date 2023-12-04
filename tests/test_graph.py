"""Unit tests for graph.py"""

import networkx as nx
from pathlib import Path

from matplotlib import pyplot as plt

from bluecellulab import SSim
from bluecellulab.graph import build_graph, plot_graph
from bluecellulab.circuit import CellId

script_dir = Path(__file__).parent


class TestGraph():
    """Test the graph.py module."""
    def setup_method(self):
        """Set up the test environment."""
        circuit_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/simulation_config_hypamp.json"
        )
        self.sim = SSim(circuit_path)
        dstut_cells = [('NodeA', 0), ('NodeA', 1), ('NodeB', 0), ('NodeB', 1)]

        self.sim.instantiate_gids(dstut_cells, add_stimuli=True, add_synapses=True)
        t_stop = 50.0
        self.sim.run(t_stop)

        # Call the build_graph function
        self.graph = build_graph(self.sim.cells)

        self.expected_edges = [
            (CellId(population_name='NodeA', id=0), CellId(population_name='NodeB', id=0)),
            (CellId(population_name='NodeA', id=0), CellId(population_name='NodeA', id=1)),
            (CellId(population_name='NodeA', id=1), CellId(population_name='NodeA', id=0)),
            (CellId(population_name='NodeA', id=1), CellId(population_name='NodeB', id=0)),
            (CellId(population_name='NodeA', id=1), CellId(population_name='NodeB', id=1)),
            (CellId(population_name='NodeB', id=0), CellId(population_name='NodeB', id=1)),
            (CellId(population_name='NodeB', id=0), CellId(population_name='NodeA', id=0)),
            (CellId(population_name='NodeB', id=1), CellId(population_name='NodeA', id=0)),
            (CellId(population_name='NodeB', id=1), CellId(population_name='NodeB', id=0)),
        ]

    def test_graph_type(self):
        """Test if the graph is an instance of nx.DiGraph."""
        assert isinstance(self.graph, nx.DiGraph)

    def test_number_of_edges(self):
        """Test if the graph has the correct number of edges."""
        expected_number_of_edges = 9
        assert len(self.graph.edges) == expected_number_of_edges

    def test_existence_of_specific_edges(self):
        """Test for the existence of specific edges."""
        for edge in self.expected_edges:
            assert self.graph.has_edge(*edge), f"Edge {edge} is missing in the graph."

    def test_no_unexpected_edges(self):
        """Test if the graph does not have unexpected edges."""
        all_edges = set(self.graph.edges)
        assert all_edges.issubset(set(self.expected_edges)), "There are unexpected edges in the graph."

    def test_correct_edge_attributes(self):
        """Test if the edges have the correct weight."""
        edge = (CellId(population_name='NodeA', id=0), CellId(population_name='NodeB', id=0))
        expected_weight = 0.37
        actual_weight = self.graph.edges[edge]['weight']
        assert abs(expected_weight - actual_weight) <= 0.01, f"Edge {edge} has incorrect weight: expected {expected_weight}, found {actual_weight}."

    def test_plot_graph(self):
        """Test the plot_graph function"""
        plot_graph(self.graph)
        assert plt.fignum_exists(1), "No figure is created."
