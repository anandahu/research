"""Digital Twin mesh network model - uses radio maps to derive link quality."""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import RSS_REFERENCE


class MeshDigitalTwin:
    """
    Digital Twin of mesh network using DT radio maps for signal quality.
    
    Maps radio map RSS values to link weights (latency/capacity).
    """
    
    def __init__(self, graph: nx.Graph, scaled_positions: Dict[int, Tuple[int, int]]):
        """
        Initialize Digital Twin.
        
        Args:
            graph: NetworkX graph from BRITE (50 nodes, 100 edges)
            scaled_positions: Node positions in DT coordinates (0-40)
        """
        self.graph = graph.copy()
        self.scaled_positions = scaled_positions
        
    def update_link_weights(self, radio_map: np.ndarray):
        """
        Update edge weights using RSS from radio map.
        
        For each edge (u, v):
        - latency = f(RSS at v position) 
        - capacity = g(RSS at v position)
        
        Args:
            radio_map: (41, 41) array of RSS values in dBm
        """
        for u, v in self.graph.edges():
            # Get position of destination node
            pos_v = self.scaled_positions[v]
            pos_u = self.scaled_positions[u]
            
            # Get RSS at both endpoints (use average)
            rss_at_v = radio_map[pos_v[0], pos_v[1]]
            rss_at_u = radio_map[pos_u[0], pos_u[1]]
            avg_rss = (rss_at_v + rss_at_u) / 2
            
            # Convert RSS to latency (lower RSS = higher latency)
            latency = self._rss_to_latency(avg_rss)
            
            # Convert RSS to capacity (higher RSS = higher capacity)
            capacity = self._rss_to_capacity(avg_rss)
            
            self.graph[u][v]['weight'] = latency
            self.graph[u][v]['capacity'] = capacity
            
    def _rss_to_latency(self, rss: float) -> float:
        """
        Convert RSS (dBm) to latency.
        
        Higher RSS = lower latency (better link)
        Uses sigmoid: latency ∈ (0, 1)
        """
        # Normalize: -100 dBm -> ~1 latency, -50 dBm -> ~0 latency
        normalized = (rss - RSS_REFERENCE) / 30.0
        latency = 1.0 / (1.0 + np.exp(normalized))
        return max(0.01, min(0.99, latency))
    
    def _rss_to_capacity(self, rss: float) -> float:
        """
        Convert RSS (dBm) to capacity.
        
        Higher RSS = higher capacity (better link)
        """
        # Normalize to 0-1 range
        capacity = (rss + 100) / 50.0
        return max(0.01, min(1.0, capacity))
    
    def get_node_rss(self, radio_map: np.ndarray) -> Dict[int, float]:
        """Get RSS value for each node from radio map."""
        rss_values = {}
        for node_id, pos in self.scaled_positions.items():
            rss_values[node_id] = radio_map[pos[0], pos[1]]
        return rss_values
    
    def get_expected_rss(self, node_id: int, radio_map: np.ndarray) -> float:
        """Get expected RSS at a node based on DT model."""
        pos = self.scaled_positions[node_id]
        return radio_map[pos[0], pos[1]]
    
    def compute_link_quality(self, radio_map: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Compute link quality metric for all edges.
        
        Returns dict of (u, v) -> quality (0-1, higher is better)
        """
        qualities = {}
        for u, v in self.graph.edges():
            pos_v = self.scaled_positions[v]
            pos_u = self.scaled_positions[u]
            rss_v = radio_map[pos_v[0], pos_v[1]]
            rss_u = radio_map[pos_u[0], pos_u[1]]
            avg_rss = (rss_v + rss_u) / 2
            
            # Quality: 0 at -100 dBm, 1 at -40 dBm
            quality = max(0.0, min(1.0, (avg_rss + 100) / 60.0))
            qualities[(u, v)] = quality
            qualities[(v, u)] = quality
            
        return qualities


if __name__ == "__main__":
    # Test digital twin
    import sys
    sys.path.insert(0, '..')
    
    print("Testing MeshDigitalTwin...")
    
    # Create simple test graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.5, capacity=0.5)
    G.add_edge(1, 2, weight=0.5, capacity=0.5)
    
    positions = {0: (10, 10), 1: (20, 20), 2: (30, 30)}
    
    dt = MeshDigitalTwin(G, positions)
    
    # Create test radio map (higher RSS in center)
    radio_map = np.full((41, 41), -80.0)
    radio_map[15:25, 15:25] = -50.0  # Strong signal in center
    
    dt.update_link_weights(radio_map)
    
    print(f"Edge (0,1) - latency: {dt.graph[0][1]['weight']:.3f}, capacity: {dt.graph[0][1]['capacity']:.3f}")
    print(f"Edge (1,2) - latency: {dt.graph[1][2]['weight']:.3f}, capacity: {dt.graph[1][2]['capacity']:.3f}")
