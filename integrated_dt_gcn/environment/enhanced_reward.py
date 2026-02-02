"""Enhanced reward function with jamming awareness."""

import networkx as nx
from typing import Tuple, Set, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import LAMBDA_JAM, ALPHA_ANOMALY, RESILIENCE_BONUS


def compute_base_reward(graph: nx.Graph, target: int, path: list) -> Tuple[float, bool]:
    """
    Compute base routing reward (from original RP15).
    
    Returns:
        (reward, done) tuple
        
    Rewards:
        +1.01: Reached target via optimal hop
        -1.51: Reached target via suboptimal hop
        (d_old - d_new): Progress toward target
        -1: Moved away from target
    """
    if len(path) < 2:
        return 0.0, False
        
    current = path[-1]
    previous = path[-2]
    
    # Check if reached target
    if current == target:
        # Check if this was optimal (direct to target)
        try:
            optimal_dist = nx.astar_path_length(graph, previous, target, weight='weight')
            actual_dist = graph[previous][current]['weight']
            if abs(optimal_dist - actual_dist) < 0.01:
                return 1.01, True
            else:
                return -1.51, True
        except:
            return 1.01, True
    
    # Check for timeout (path too long)
    if len(path) > 10 * len(graph.nodes):
        return -1.0, True
    
    # Progress reward
    try:
        old_dist = nx.astar_path_length(graph, previous, target, weight='weight')
        new_dist = nx.astar_path_length(graph, current, target, weight='weight')
        progress = old_dist - new_dist
        if progress > 0:
            return progress, False
        else:
            return -1.0, False
    except:
        return -1.0, False


def compute_enhanced_reward(
    graph: nx.Graph,
    target: int,
    path: list,
    jammed_nodes: Set[int],
    anomaly_scores: Dict[int, float],
    stepped_on_jammed: bool = False
) -> Tuple[float, bool]:
    """
    Enhanced reward with jamming penalties and resilience bonuses.
    
    Reward = R_base × (1 - α × anomaly) - λ × jammed + resilience_bonus
    
    Args:
        graph: Network graph
        target: Target node
        path: Current path taken
        jammed_nodes: Set of jammed node IDs
        anomaly_scores: Dict node_id -> anomaly score (0-1)
        stepped_on_jammed: Whether stepped on jammed node this step
        
    Returns:
        (reward, done) tuple
    """
    # Get base reward
    base_reward, done = compute_base_reward(graph, target, path)
    
    current = path[-1]
    
    # Get anomaly score for current node
    anomaly = anomaly_scores.get(current, 0.0)
    
    # Dampen base reward by anomaly
    dampened_reward = base_reward * (1 - ALPHA_ANOMALY * anomaly)
    
    # Apply jamming penalty if stepped on jammed node
    jam_penalty = 0.0
    if stepped_on_jammed or current in jammed_nodes:
        jam_penalty = -LAMBDA_JAM
    
    # Resilience bonus: reached target while avoiding jammed nodes
    resilience = 0.0
    if done and base_reward > 0:
        path_jammed = any(n in jammed_nodes for n in path)
        if jammed_nodes and not path_jammed:
            resilience = RESILIENCE_BONUS
    
    total_reward = dampened_reward + jam_penalty + resilience
    
    return total_reward, done


def compute_path_length(graph: nx.Graph, path: tuple) -> float:
    """Compute total latency (weight) along path."""
    total = 0.0
    for i in range(len(path) - 1):
        try:
            total += graph[path[i]][path[i+1]]['weight']
        except:
            total += 1.0
    return total


def compute_flow_value(graph: nx.Graph, path: tuple) -> float:
    """Compute minimum capacity (bottleneck) along path."""
    min_cap = float('inf')
    for i in range(len(path) - 1):
        try:
            cap = graph[path[i]][path[i+1]]['capacity']
            min_cap = min(min_cap, cap)
        except:
            pass
    return min_cap if min_cap < float('inf') else 0.0


if __name__ == "__main__":
    # Test reward functions
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.5, capacity=0.5)
    G.add_edge(1, 2, weight=0.5, capacity=0.5)
    G.add_edge(0, 2, weight=1.2, capacity=0.3)
    
    # Test base reward
    r, d = compute_base_reward(G, 2, [0, 1])
    print(f"Progress 0->1 toward 2: reward={r:.2f}, done={d}")
    
    r, d = compute_base_reward(G, 2, [0, 1, 2])
    print(f"Reached 0->1->2: reward={r:.2f}, done={d}")
    
    # Test enhanced reward with jamming
    jammed = {1}
    anomaly = {0: 0.0, 1: 0.8, 2: 0.0}
    
    r, d = compute_enhanced_reward(G, 2, [0, 1], jammed, anomaly, stepped_on_jammed=True)
    print(f"Stepped on jammed node 1: reward={r:.2f}")
    
    r, d = compute_enhanced_reward(G, 2, [0, 2], jammed, anomaly, stepped_on_jammed=False)
    print(f"Avoided jammed, reached target: reward={r:.2f}")
