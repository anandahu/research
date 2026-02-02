"""Anomaly detection bridge - converts DT measurement diffs to node anomaly scores."""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import DT_MAP_SIZE


class AnomalyBridge:
    """
    Converts Digital Twin measurement differences to per-node anomaly scores.
    
    The DT provides 25 measurement points with diff values (expected - actual RSS).
    This bridge maps these to anomaly scores for each of the 50 mesh nodes.
    """
    
    def __init__(self):
        """Initialize anomaly bridge."""
        self.threshold = None
        self.mean = None
        self.std = None
        self.is_trained = False
        
    def train(self, normal_measurements: List[np.ndarray]):
        """
        Train threshold from normal (non-jammed) measurement samples.
        
        Uses 2-sigma threshold: anomaly if diff > mean + 2*std
        
        Args:
            normal_measurements: List of (25,) arrays from non-jammed scenarios
        """
        # Concatenate all absolute diffs
        all_diffs = np.concatenate([np.abs(m) for m in normal_measurements])
        
        self.mean = np.mean(all_diffs)
        self.std = np.std(all_diffs)
        self.threshold = self.mean + 2 * self.std  # 2-sigma threshold
        
        self.is_trained = True
        print(f"AnomalyBridge trained: mean={self.mean:.2f}, std={self.std:.2f}, threshold={self.threshold:.2f}")
        
    def compute_measurement_scores(self, meas_diff: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores from measurement difference array.
        
        Args:
            meas_diff: (25,) array of DT - actual RSS differences
            
        Returns:
            (25,) array of anomaly scores (0-1)
        """
        if not self.is_trained:
            raise ValueError("AnomalyBridge not trained. Call train() first.")
            
        abs_diff = np.abs(meas_diff)
        
        # Normalize to 0-1 using sigmoid-like scaling
        # Score = 0 if diff <= mean, increasing toward 1 as diff increases
        scores = np.zeros_like(abs_diff)
        above_mean = abs_diff > self.mean
        scores[above_mean] = np.minimum(1.0, (abs_diff[above_mean] - self.mean) / (2 * self.threshold))
        
        return scores
    
    def map_to_nodes(self, meas_scores: np.ndarray, 
                     meas_x: List[int], meas_y: List[int],
                     node_positions: Dict[int, Tuple[int, int]]) -> Dict[int, float]:
        """
        Map measurement point scores to mesh node scores.
        
        Uses nearest measurement point for each node.
        
        Args:
            meas_scores: (25,) array of measurement anomaly scores
            meas_x, meas_y: Measurement point coordinates (DT coords)
            node_positions: Dict of node_id -> (x, y) in DT coordinates
            
        Returns:
            Dict of node_id -> anomaly score (0-1)
        """
        meas_points = np.array(list(zip(meas_x, meas_y)))  # (25, 2)
        node_scores = {}
        
        for node_id, pos in node_positions.items():
            # Find nearest measurement point
            distances = np.sqrt((meas_points[:, 0] - pos[0])**2 + 
                               (meas_points[:, 1] - pos[1])**2)
            nearest_idx = np.argmin(distances)
            
            # Assign score from nearest measurement point
            # Weight by inverse distance (closer = more relevant)
            min_dist = distances[nearest_idx]
            if min_dist < 1.0:
                node_scores[node_id] = meas_scores[nearest_idx]
            else:
                # Decay score with distance
                decay = np.exp(-min_dist / 10.0)
                node_scores[node_id] = meas_scores[nearest_idx] * decay
                
        return node_scores
    
    def compute_node_scores(self, meas_diff: np.ndarray,
                           meas_x: List[int], meas_y: List[int],
                           node_positions: Dict[int, Tuple[int, int]]) -> Dict[int, float]:
        """
        Full pipeline: measurement diff -> node anomaly scores.
        
        Args:
            meas_diff: (25,) array of DT - actual differences
            meas_x, meas_y: Measurement coordinates
            node_positions: Node positions in DT coordinates
            
        Returns:
            Dict of node_id -> anomaly score (0-1)
        """
        meas_scores = self.compute_measurement_scores(meas_diff)
        return self.map_to_nodes(meas_scores, meas_x, meas_y, node_positions)


class JammerDetector:
    """
    Detect which nodes are likely jammed based on jammer positions.
    """
    
    def __init__(self, jam_radius: float = 10.0):
        """
        Args:
            jam_radius: Radius of jamming effect in DT coordinates
        """
        self.jam_radius = jam_radius
        
    def get_jammed_nodes(self, jammer_positions: List[Tuple[int, int]],
                         node_positions: Dict[int, Tuple[int, int]]) -> set:
        """
        Get set of nodes affected by jammers.
        
        Args:
            jammer_positions: List of jammer (x, y) in DT coordinates
            node_positions: Dict of node_id -> (x, y) in DT coordinates
            
        Returns:
            Set of jammed node IDs
        """
        if not jammer_positions:
            return set()
            
        jammed = set()
        for node_id, pos in node_positions.items():
            for jam_pos in jammer_positions:
                dist = np.sqrt((pos[0] - jam_pos[0])**2 + (pos[1] - jam_pos[1])**2)
                if dist <= self.jam_radius:
                    jammed.add(node_id)
                    break
                    
        return jammed
    
    def get_jam_probabilities(self, jammer_positions: List[Tuple[int, int]],
                              node_positions: Dict[int, Tuple[int, int]]) -> Dict[int, float]:
        """
        Get per-node jamming probability (soft version of get_jammed_nodes).
        
        Returns probability based on distance to nearest jammer.
        """
        if not jammer_positions:
            return {n: 0.0 for n in node_positions}
            
        probs = {}
        for node_id, pos in node_positions.items():
            min_dist = float('inf')
            for jam_pos in jammer_positions:
                dist = np.sqrt((pos[0] - jam_pos[0])**2 + (pos[1] - jam_pos[1])**2)
                min_dist = min(min_dist, dist)
            
            # Sigmoid decay based on distance
            if min_dist <= self.jam_radius:
                probs[node_id] = 1.0
            else:
                probs[node_id] = np.exp(-(min_dist - self.jam_radius) / 5.0)
                
        return probs


if __name__ == "__main__":
    # Test anomaly bridge
    print("Testing AnomalyBridge...")
    
    bridge = AnomalyBridge()
    
    # Simulate normal measurements
    normal = [np.random.normal(0, 2, 25) for _ in range(100)]
    bridge.train(normal)
    
    # Test with normal sample
    normal_scores = bridge.compute_measurement_scores(normal[0])
    print(f"Normal sample: max score = {normal_scores.max():.3f}")
    
    # Test with anomalous sample (high diffs)
    anomaly = np.random.normal(10, 3, 25)  # Much higher diffs
    anomaly_scores = bridge.compute_measurement_scores(anomaly)
    print(f"Anomaly sample: max score = {anomaly_scores.max():.3f}")
