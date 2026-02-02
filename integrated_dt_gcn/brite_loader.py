"""BRITE topology file loader."""

import networkx as nx
from typing import Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import DT_MAP_SIZE, BRITE_COORD_RANGE, COORD_SCALE


class BRITELoader:
    """Load and parse 50nodes.brite topology file."""
    
    def __init__(self, filepath: str):
        """
        Initialize BRITE loader.
        
        Args:
            filepath: Path to .brite file
        """
        self.filepath = filepath
        self.graph = None
        
    def load_graph(self) -> nx.Graph:
        """
        Parse BRITE file and create NetworkX graph.
        
        Returns:
            NetworkX graph with 50 nodes and 100 edges
            Each node has 'pos' attribute (x, y)
            Each edge has 'weight' (latency) and 'capacity' attributes
        """
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip()
        # "Topology: ( 50 Nodes, 100 Edges )"
        parts = header.split()
        num_nodes = int(parts[2])
        num_edges = int(parts[4])
        
        self.graph = nx.Graph()
        
        # Find Nodes section (starts at line 4)
        node_start = 4
        for i in range(num_nodes):
            line = lines[node_start + i].strip()
            parts = line.split('\t')
            node_id = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            self.graph.add_node(node_id, pos=(x, y))
        
        # Find Edges section
        edge_start = node_start + num_nodes + 3  # Skip blank lines and "Edges:" header
        for i in range(num_edges):
            line = lines[edge_start + i].strip()
            parts = line.split('\t')
            # Format: edge_id, src, dst, weight, capacity, ...
            src = int(parts[1])
            dst = int(parts[2])
            weight = float(parts[3])  # Latency
            capacity = float(parts[4])  # Bandwidth
            self.graph.add_edge(src, dst, weight=weight, capacity=capacity)
        
        return self.graph
    
    def get_node_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get dictionary of node positions."""
        if self.graph is None:
            self.load_graph()
        return {n: self.graph.nodes[n]['pos'] for n in self.graph.nodes()}
    
    def scale_position_to_dt(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Scale BRITE coordinates (0-99) to DT radio map coordinates (0-40).
        
        Args:
            pos: (x, y) position in BRITE coordinates
            
        Returns:
            (x, y) position in DT coordinates (0-40)
        """
        x_dt = min(DT_MAP_SIZE - 1, int(pos[0] * COORD_SCALE))
        y_dt = min(DT_MAP_SIZE - 1, int(pos[1] * COORD_SCALE))
        return (x_dt, y_dt)
    
    def get_scaled_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get node positions scaled to DT coordinates."""
        positions = self.get_node_positions()
        return {n: self.scale_position_to_dt(p) for n, p in positions.items()}


if __name__ == "__main__":
    # Test loading
    loader = BRITELoader("../RP15/50nodes.brite")
    G = loader.load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test position scaling
    positions = loader.get_node_positions()
    scaled = loader.get_scaled_positions()
    print(f"Node 0: BRITE {positions[0]} -> DT {scaled[0]}")
    print(f"Node 5: BRITE {positions[5]} -> DT {scaled[5]}")
