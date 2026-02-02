"""Hybrid environment: BRITE topology + DT signal data."""

import gym
import torch
import numpy as np
import networkx as nx
import random
from typing import Tuple, Dict, Set, List, Any
from gym.spaces import MultiDiscrete, Discrete
from torch_geometric.data import Data
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import (
    device, NUM_NODES, NODE_FEATURE_DIM, DT_MAP_SIZE, COORD_SCALE
)
from integrated_dt_gcn.brite_loader import BRITELoader
from integrated_dt_gcn.dataset_loader import DTDatasetLoader
from integrated_dt_gcn.digital_twin.mesh_twin import MeshDigitalTwin
from integrated_dt_gcn.digital_twin.anomaly_bridge import AnomalyBridge, JammerDetector
from integrated_dt_gcn.environment.enhanced_reward import compute_enhanced_reward


class HybridEnv(gym.Env):
    """
    Hybrid environment using:
    - BRITE topology (50 nodes, 100 edges)
    - DT radio maps for link weights
    - DT measurements for anomaly scores
    - DT jammer labels for ground truth
    
    Node features (7-dim):
        [is_source, is_dest, avg_latency, avg_bandwidth, anomaly_score, jam_prob, neighbor_jam_avg]
    """
    
    def __init__(self, brite_path: str, dt_loader: DTDatasetLoader,
                 anomaly_bridge: AnomalyBridge, dataset_nr: int = 0):
        """
        Initialize hybrid environment.
        
        Args:
            brite_path: Path to 50nodes.brite
            dt_loader: Loaded DTDatasetLoader
            anomaly_bridge: Trained AnomalyBridge
            dataset_nr: Which DT dataset to use (0-5)
        """
        super().__init__()
        
        # Load BRITE graph
        self.brite_loader = BRITELoader(brite_path)
        self.base_graph = self.brite_loader.load_graph()
        self.graph = self.base_graph.copy()
        
        # Get scaled positions
        self.scaled_positions = self.brite_loader.get_scaled_positions()
        
        # DT components
        self.dt_loader = dt_loader
        self.anomaly_bridge = anomaly_bridge
        self.jammer_detector = JammerDetector(jam_radius=10.0)
        self.dataset_nr = dataset_nr
        
        # Pre-load scenario indices
        self.jammed_indices = dt_loader.get_jammed_indices(dataset_nr)
        self.normal_indices = dt_loader.get_normal_indices(dataset_nr)
        
        # Digital Twin
        self.mesh_twin = MeshDigitalTwin(self.graph, self.scaled_positions)
        
        # State
        self.num_nodes = self.graph.number_of_nodes()
        self.current_scenario_idx = 0
        self.source = -1
        self.target = -1
        self.current_node = -1
        self.path = []
        self.jammed_nodes: Set[int] = set()
        self.anomaly_scores: Dict[int, float] = {}
        self.jam_probabilities: Dict[int, float] = {}
        
        # Episode tracking
        self.episode = 0
        self.steps = 0
        
        # Gym spaces
        self.observation_space = MultiDiscrete([self.num_nodes, self.num_nodes])
        self.action_space = Discrete(self.num_nodes)
        
        # Edge data for GNN
        self.edge_index = self._build_edge_index()
        
    def _build_edge_index(self) -> torch.Tensor:
        """Build edge index tensor for PyG."""
        edges = list(self.graph.edges())
        # Add reverse edges (undirected graph)
        edge_list = edges + [(v, u) for u, v in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index.to(device)
    
    def _select_scenario(self) -> int:
        """Select scenario index (50% jammed, 50% normal)."""
        if random.random() < 0.5 and self.jammed_indices:
            return random.choice(self.jammed_indices)
        else:
            return random.choice(self.normal_indices)
    
    def _get_new_route(self) -> Tuple[int, int]:
        """Get random source-target pair with valid path."""
        nodes = list(self.graph.nodes())
        for _ in range(100):
            src = random.choice(nodes)
            tgt = random.choice(nodes)
            if src != tgt:
                try:
                    nx.shortest_path(self.graph, src, tgt)
                    return src, tgt
                except:
                    continue
        # Fallback
        return 0, self.num_nodes - 1
    
    def reset(self) -> Tuple[Data, Dict]:
        """
        Reset environment with new scenario from dataset.
        
        Returns:
            (graph_data, info_dict) where graph_data is PyG Data object
        """
        self.episode += 1
        self.steps = 0
        
        # Select new scenario
        self.current_scenario_idx = self._select_scenario()
        
        # Load radio map and update link weights
        radio_map = self.dt_loader.get_radio_map_array(self.current_scenario_idx, self.dataset_nr)
        self.mesh_twin.update_link_weights(radio_map)
        self.graph = self.mesh_twin.graph
        
        # Get jammer info
        jammer_positions = self.dt_loader.get_jammer_positions(self.current_scenario_idx, self.dataset_nr)
        # Scale jammer positions to DT coordinates
        scaled_jammer_pos = [(int(p[0] * COORD_SCALE), int(p[1] * COORD_SCALE)) for p in jammer_positions]
        
        # Detect jammed nodes
        self.jammed_nodes = self.jammer_detector.get_jammed_nodes(scaled_jammer_pos, self.scaled_positions)
        self.jam_probabilities = self.jammer_detector.get_jam_probabilities(scaled_jammer_pos, self.scaled_positions)
        
        # Compute anomaly scores
        meas_diff = self.dt_loader.get_measurement_diff(self.current_scenario_idx, self.dataset_nr)
        meas_x, meas_y = self.dt_loader.get_measurement_points(self.dataset_nr)
        self.anomaly_scores = self.anomaly_bridge.compute_node_scores(
            meas_diff, meas_x, meas_y, self.scaled_positions
        )
        
        # Setup routing
        self.source, self.target = self._get_new_route()
        self.current_node = self.source
        self.path = [self.source]
        
        # Build observation
        node_features = self._compute_node_features()
        valid_actions = self._get_valid_actions()
        
        return Data(x=node_features, edge_index=self.edge_index), {'valid_actions': valid_actions}
    
    def step(self, action: int) -> Tuple[Data, float, bool, Dict]:
        """
        Take action (move to neighbor node).
        
        Args:
            action: Node ID to move to
            
        Returns:
            (observation, reward, done, info)
        """
        self.steps += 1
        next_node = action
        
        # Validate action
        if next_node not in list(self.graph.neighbors(self.current_node)):
            # Invalid action - stayed in place with penalty
            reward = -1.0
            node_features = self._compute_node_features()
            valid_actions = self._get_valid_actions()
            return Data(x=node_features, edge_index=self.edge_index), reward, False, {'valid_actions': valid_actions}
        
        # Move to next node
        self.path.append(next_node)
        stepped_on_jammed = next_node in self.jammed_nodes
        self.current_node = next_node
        
        # Compute enhanced reward
        reward, done = compute_enhanced_reward(
            self.graph,
            self.target,
            self.path,
            self.jammed_nodes,
            self.anomaly_scores,
            stepped_on_jammed
        )
        
        # Build observation
        node_features = self._compute_node_features()
        valid_actions = self._get_valid_actions()
        
        return Data(x=node_features, edge_index=self.edge_index), reward, done, {'valid_actions': valid_actions}
    
    def _compute_node_features(self) -> torch.Tensor:
        """
        Compute 7-dimensional node features.
        
        Features:
            0: is_source (1 if current source)
            1: is_dest (1 if target)
            2: avg_latency (mean edge weight to neighbors)
            3: avg_bandwidth (mean edge capacity to neighbors)
            4: anomaly_score (from DT)
            5: jam_probability (from jammer detector)
            6: neighbor_jam_avg (mean jam_prob of neighbors)
        """
        x = torch.zeros((self.num_nodes, NODE_FEATURE_DIM), device=device)
        
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            
            # Feature 0: is_source (current position)
            x[node, 0] = 1.0 if node == self.current_node else 0.0
            
            # Feature 1: is_dest
            x[node, 1] = 1.0 if node == self.target else 0.0
            
            # Feature 2 & 3: avg latency and bandwidth
            if neighbors:
                x[node, 2] = np.mean([self.graph[node][n]['weight'] for n in neighbors])
                x[node, 3] = np.mean([self.graph[node][n]['capacity'] for n in neighbors])
            
            # Feature 4: anomaly score
            x[node, 4] = self.anomaly_scores.get(node, 0.0)
            
            # Feature 5: jam probability
            x[node, 5] = self.jam_probabilities.get(node, 0.0)
            
            # Feature 6: neighbor jam average
            if neighbors:
                x[node, 6] = np.mean([self.jam_probabilities.get(n, 0.0) for n in neighbors])
        
        return x
    
    def _get_valid_actions(self) -> torch.Tensor:
        """Get mask of valid actions (neighbor nodes)."""
        valid = torch.zeros((1, self.num_nodes), dtype=torch.bool, device=device)
        neighbors = list(self.graph.neighbors(self.current_node))
        for n in neighbors:
            valid[0, n] = True
        return valid
    
    def get_scenario_info(self) -> Dict:
        """Get info about current scenario."""
        return {
            'scenario_idx': self.current_scenario_idx,
            'has_jammer': bool(self.jammed_nodes),
            'num_jammed_nodes': len(self.jammed_nodes),
            'source': self.source,
            'target': self.target
        }


if __name__ == "__main__":
    print("Testing HybridEnv...")
    
    # Initialize components
    loader = DTDatasetLoader("../RP12_paper/datasets")
    
    # Train anomaly bridge
    bridge = AnomalyBridge()
    normal_meas = loader.get_normal_measurements(0)[:1000]
    bridge.train(normal_meas)
    
    # Create environment
    env = HybridEnv("../RP15/50nodes.brite", loader, bridge)
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.x.shape}")
    print(f"Valid actions: {info['valid_actions'].sum().item()} neighbors")
    print(f"Scenario info: {env.get_scenario_info()}")
    
    # Test step
    valid_neighbors = info['valid_actions'][0].nonzero().squeeze().tolist()
    if isinstance(valid_neighbors, int):
        valid_neighbors = [valid_neighbors]
    action = random.choice(valid_neighbors)
    print(f"Taking action: {action}")
    
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.3f}, Done: {done}")
