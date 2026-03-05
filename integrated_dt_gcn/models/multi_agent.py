"""
Multi-Agent DQN with GCN + Digital Twin awareness.

RP15-style: one agent per node, each with its own GCN policy/target network.
Each node's agent learns to forward packets to the best neighbor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import numpy as np
from tqdm import tqdm
from itertools import count
from collections import namedtuple
from torch_geometric.loader import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import (
    device, NODE_FEATURE_DIM, EPS_START, EPS_END,
    BATCH_SIZE, GAMMA, LEARNING_RATE,
    MA_EPS_DECAY, MA_REPLAY_SIZE, MA_TARGET_UPDATE, MA_BATCH_SIZE
)
from integrated_dt_gcn.models.gat_dt_aware import GAT_DTAware, ReplayMemory

# Soft target update rate
TAU = 0.005

# Shared replay pool capacity (all 50 agents contribute to this)
SHARED_POOL_SIZE = 50000
# Fraction of a batch to top-up from the shared pool when an agent's own buffer is sparse
SHARED_SAMPLE_RATIO = 0.4


class SharedReplayMemory:
    """
    Global experience pool shared across all node agents.

    Each stored entry includes the acting agent's `num_neighbors` so that
    an agent only samples transitions whose action index is valid for its
    own output dimension (avoids index-out-of-bounds in Q-value gather).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []   # list of (transition, num_neighbors)
        self.position = 0

    def push(self, num_neighbors: int, *args):
        """Save a transition tagged with the acting agent's num_neighbors."""
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        entry = (Transition(*args), num_neighbors)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample_for_agent(self, batch_size: int, num_neighbors: int):
        """
        Sample transitions compatible with `num_neighbors`.
        Returns up to `batch_size` transitions (may be fewer if pool is sparse).
        """
        compatible = [t for t, n in self.memory if t is not None and n == num_neighbors]
        if len(compatible) < batch_size:
            return []
        return random.sample(compatible, batch_size)

    def __len__(self):
        return len(self.memory)


class NodeAgent:
    """
    A single node's DQN agent with GAT_DTAware model.
    
    Each node has its own:
    - Policy network (GAT_DTAware with out_dim = num_neighbors)
    - Target network (copy of policy)
    - Replay memory
    - Optimizer
    - Step counter for epsilon-greedy
    """
    
    def __init__(self, node_id: int, num_nodes: int, num_neighbors: int,
                 neighbor_ids: list, node_feat_dim: int = NODE_FEATURE_DIM):
        """
        Args:
            node_id: This node's ID in the graph
            num_nodes: Total nodes in graph (50)
            num_neighbors: Number of neighbors for this node
            neighbor_ids: List of neighbor node IDs
            node_feat_dim: Input feature dimension (7)
        """
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.neighbor_ids = sorted(neighbor_ids)  # Sorted for consistent indexing
        
        # Create policy and target networks
        # Output dim = num_neighbors (Q-value per neighbor)
        self.policy_net = GAT_DTAware(
            num_nodes=num_nodes,
            out_dim=num_neighbors,
            node_feat_dim=node_feat_dim
        ).to(device)
        
        self.target_net = GAT_DTAware(
            num_nodes=num_nodes,
            out_dim=num_neighbors,
            node_feat_dim=node_feat_dim
        ).to(device)
        
        # Sync target with policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MA_REPLAY_SIZE)
        self.steps_done = 0

    def select_action(self, state_loader, eps_threshold: float) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state_loader: DataLoader with PyG Data (graph observation)
            eps_threshold: Current epsilon for exploration
            
        Returns:
            Neighbor INDEX (0 to num_neighbors-1), NOT node ID
        """
        sample = random.random()
        
        if sample > eps_threshold:
            # Exploit: use policy network
            with torch.no_grad():
                self.policy_net.eval()
                # Mask: all neighbors are valid
                mask = torch.ones((1, self.num_neighbors), dtype=torch.bool, device=device)
                q_values = self.policy_net((state_loader, mask))
                action_idx = q_values.max(1)[1].item()
                self.policy_net.train()
                return action_idx
        else:
            # Explore: random neighbor
            return random.randint(0, self.num_neighbors - 1)
    
    def neighbor_idx_to_node_id(self, idx: int) -> int:
        """Convert neighbor index to actual node ID."""
        return self.neighbor_ids[idx]
    
    def node_id_to_neighbor_idx(self, node_id: int) -> int:
        """Convert node ID to neighbor index."""
        return self.neighbor_ids.index(node_id)

    def optimize_model(self, shared_pool=None):
        """
        Perform one optimization step from replay memory.

        If shared_pool is provided:
        - Agents with enough own experience: use (1-SHARED_SAMPLE_RATIO) of
          the batch from own memory + SHARED_SAMPLE_RATIO from shared pool.
        - Agents below MA_BATCH_SIZE in own memory: try to fill the full
          batch from the shared pool so they can still train early on.
        """
        own_count = len(self.memory)
        shared_count = len(shared_pool) if shared_pool else 0

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        if own_count >= MA_BATCH_SIZE:
            # Has enough own experience — blend with shared pool
            n_shared = int(MA_BATCH_SIZE * SHARED_SAMPLE_RATIO) if shared_pool else 0
            n_own = MA_BATCH_SIZE - n_shared
            own_transitions = self.memory.sample(n_own)
            shared_transitions = shared_pool.sample_for_agent(n_shared, self.num_neighbors) if (shared_pool and n_shared > 0) else []
            transitions = own_transitions + shared_transitions
            if len(transitions) < MA_BATCH_SIZE:
                return  # Not enough data even with top-up
        elif shared_pool and len(shared_pool.sample_for_agent(MA_BATCH_SIZE, self.num_neighbors)) == MA_BATCH_SIZE:
            # Own buffer too small — borrow from shared pool (same num_neighbors only)
            transitions = shared_pool.sample_for_agent(MA_BATCH_SIZE, self.num_neighbors)
        else:
            return  # Neither own nor shared pool has enough compatible data yet

        batch = Transition(*zip(*transitions))
        
        # Build masks for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        
        # Collect non-final next state graphs
        non_final_next_graphs = [s for s in batch.next_state if s is not None]
        
        if non_final_next_graphs:
            non_final_loader = DataLoader(non_final_next_graphs,
                                          batch_size=len(non_final_next_graphs),
                                          shuffle=False)
            non_final_mask_tensor = torch.ones(
                (len(non_final_next_graphs), self.num_neighbors),
                dtype=torch.bool, device=device
            )
        
        # State batch
        state_graphs = list(batch.state)
        state_loader = DataLoader(state_graphs, batch_size=len(state_graphs), shuffle=False)
        state_mask = torch.ones(
            (len(state_graphs), self.num_neighbors),
            dtype=torch.bool, device=device
        )
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Q(s, a)
        self.policy_net.train()
        state_action_values = self.policy_net(
            (state_loader, state_mask)
        ).gather(1, action_batch)
        
        # V(s') = max_a Q_target(s', a)
        next_state_values = torch.zeros(MA_BATCH_SIZE, device=device)
        if non_final_next_graphs:
            next_state_values[non_final_mask] = self.target_net(
                (non_final_loader, non_final_mask_tensor)
            ).max(1)[0].detach()
        
        # Expected Q values
        expected = (next_state_values * GAMMA) + reward_batch
        
        # Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sync_target(self):
        """Soft update target network: target = tau * policy + (1-tau) * target."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)


class MultiAgentCoordinator:
    """
    RP15-style multi-agent system: one NodeAgent per graph node.
    
    Each node has its own GAT_DTAware model that outputs Q-values
    for its neighbors. During routing, the current node's agent
    picks the next hop.
    """
    
    def __init__(self, env):
        """
        Initialize 50 node agents from environment graph.
        
        Args:
            env: HybridEnv instance
        """
        self.env = env
        self.num_nodes = env.num_nodes
        self.nodes = list(env.graph.nodes())
        self.agents = {}
        
        # Total params tracker
        total_params = 0
        
        # Create one agent per node
        for node in self.nodes:
            neighbors = sorted(list(env.graph.neighbors(node)))
            num_neighbors = len(neighbors)
            
            if num_neighbors == 0:
                continue  # Skip isolated nodes
                
            agent = NodeAgent(
                node_id=node,
                num_nodes=self.num_nodes,
                num_neighbors=num_neighbors,
                neighbor_ids=neighbors,
                node_feat_dim=NODE_FEATURE_DIM
            )
            self.agents[node] = agent
            total_params += sum(p.numel() for p in agent.policy_net.parameters())
        
        print(f"[MultiAgent] Created {len(self.agents)} node agents")
        print(f"[MultiAgent] Total parameters: {total_params:,}")
        
        # Episode counter for epsilon decay (NOT steps - that was the bug)
        self.episode_count = 0
        self.global_steps = 0

        # Shared experience pool — all agents contribute and can sample from here
        self.shared_pool = SharedReplayMemory(SHARED_POOL_SIZE)
        
        # Metrics
        self.metrics = {
            'loss': [],
            'reward': [],
            'path_length': [],
            'eps_reward': [],
            'jammed_steps': [],
            'success_rate': [],
            'episode_success': [],
            'step_latency': [],
            'step_bandwidth': [],
            'step_pdr': [],
            'eps_total_latency': [],
            'eps_avg_bandwidth': [],
            'eps_avg_pdr': []
        }
    
    def _get_eps_threshold(self) -> float:
        """Compute current epsilon from EPISODE count (not steps)."""
        return EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.episode_count / MA_EPS_DECAY)
    
    def _make_loader(self, obs):
        """Create a DataLoader from a single observation."""
        return DataLoader([obs], batch_size=1, shuffle=False)
    
    def run(self, num_episodes: int = 1000, verbose: bool = True):
        """
        Train all agents via environment interaction.
        
        In each episode:
        1. Reset environment → get source, target
        2. Current node's agent selects action (neighbor)
        3. Environment steps → reward, next observation
        4. Store transition in current agent's memory
        5. Optimize current agent
        6. Move to next node, repeat
        
        Args:
            num_episodes: Number of training episodes
            verbose: Print progress
        """
        successes = 0
        
        iterator = tqdm(range(num_episodes), desc="MA Training") if verbose else range(num_episodes)
        
        for episode in iterator:
            obs, info = self.env.reset()
            current_node = self.env.current_node
            
            eps_reward = 0
            jammed_count = 0
            eps_latency = 0
            eps_bandwidths = []
            eps_pdrs = []
            
            for t in count():
                self.global_steps += 1
                
                # Skip if current node has no agent (isolated)
                if current_node not in self.agents:
                    break
                
                agent = self.agents[current_node]
                eps = self._get_eps_threshold()
                
                # Agent selects neighbor INDEX
                state_loader = self._make_loader(obs)
                neighbor_idx = agent.select_action(state_loader, eps)
                
                # Convert to node ID for environment
                next_node_id = agent.neighbor_idx_to_node_id(neighbor_idx)
                
                # Step environment
                next_obs, reward, done, next_info = self.env.step(next_node_id)
                
                eps_reward += reward
                self.metrics['reward'].append(reward)
                
                # Track jammed steps
                if next_node_id in self.env.jammed_nodes:
                    jammed_count += 1
                
                # Track latency/bandwidth
                prev_node = current_node
                try:
                    step_latency = self.env.graph[prev_node][next_node_id]['weight']
                    step_bandwidth = self.env.graph[prev_node][next_node_id]['capacity']
                except:
                    step_latency = 0
                    step_bandwidth = 0
                
                self.metrics['step_latency'].append(step_latency)
                self.metrics['step_bandwidth'].append(step_bandwidth)
                self.metrics['step_pdr'].append(step_bandwidth)
                eps_latency += step_latency
                eps_bandwidths.append(step_bandwidth)
                eps_pdrs.append(step_bandwidth)
                
                # Store transition in THIS agent's memory AND in the shared pool
                action_tensor = torch.tensor([[neighbor_idx]], device=device, dtype=torch.long)
                reward_tensor = torch.tensor([reward], device=device)
                next_state = next_obs if not done else None

                agent.memory.push(obs, action_tensor, next_state, reward_tensor)
                self.shared_pool.push(agent.num_neighbors, obs, action_tensor, next_state, reward_tensor)

                # Optimize this agent (shared pool used internally when needed)
                loss = agent.optimize_model(self.shared_pool)
                if loss is not None:
                    self.metrics['loss'].append(loss)
                
                # Move to next node
                obs = next_obs
                current_node = self.env.current_node
                
                if done:
                    self.metrics['path_length'].append(t + 1)
                    self.metrics['eps_reward'].append(eps_reward)
                    self.metrics['jammed_steps'].append(jammed_count)
                    self.metrics['eps_total_latency'].append(eps_latency)
                    self.metrics['eps_avg_bandwidth'].append(
                        sum(eps_bandwidths) / len(eps_bandwidths) if eps_bandwidths else 0
                    )
                    self.metrics['eps_avg_pdr'].append(
                        sum(eps_pdrs) / len(eps_pdrs) if eps_pdrs else 0
                    )
                    # Fix: use node position, not reward sign (jamming penalties
                    # can make reward < 0 even when the target was reached)
                    if self.env.current_node == self.env.target:
                        successes += 1
                        self.metrics['episode_success'].append(1)
                    else:
                        self.metrics['episode_success'].append(0)
                    break

            # Increment episode counter for epsilon decay
            self.episode_count += 1
            
            # Soft update target networks every episode
            # (tau=0.005 makes this very gradual, unlike hard copy)
            if episode % MA_TARGET_UPDATE == 0:
                for agent in self.agents.values():
                    agent.sync_target()
            
            # Progress logging
            if verbose and episode % 50 == 0 and episode > 0:
                recent_rewards = self.metrics['eps_reward'][-50:]
                success_rate = successes / (episode + 1)
                print(f"\nEpisode {episode}: avg_reward={np.mean(recent_rewards):.2f}, "
                      f"success_rate={success_rate:.2%}, eps={self._get_eps_threshold():.3f}")
        
        if verbose:
            print(f"\nTraining complete. Final success rate: {successes/num_episodes:.2%}")
    
    def test(self, num_episodes: int = 500, verbose: bool = True):
        """
        Evaluate agents using greedy policy (no exploration).
        
        Args:
            num_episodes: Number of test episodes
            verbose: Print results
        """
        good = 0
        bad = 0
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            current_node = self.env.current_node
            
            for t in count():
                if current_node not in self.agents:
                    break
                
                agent = self.agents[current_node]
                state_loader = self._make_loader(obs)
                
                # Greedy: epsilon = 0
                with torch.no_grad():
                    agent.policy_net.eval()
                    mask = torch.ones((1, agent.num_neighbors), dtype=torch.bool, device=device)
                    q_values = agent.policy_net((state_loader, mask))
                    neighbor_idx = q_values.max(1)[1].item()
                
                next_node_id = agent.neighbor_idx_to_node_id(neighbor_idx)
                obs, reward, done, info = self.env.step(next_node_id)
                current_node = self.env.current_node
                
                if done:
                    if reward > 0:
                        good += 1
                    else:
                        bad += 1
                    break
        
        rate = good / float(good + bad) if (good + bad) > 0 else 0
        if verbose:
            print(f"[MultiAgent Test] Routed: {rate:.2%}  ({good} good, {bad} bad)")
        return rate
    
    def save(self, path: str):
        """Save all 50 agents' weights to a single file."""
        checkpoint = {
            'global_steps': self.global_steps,
            'agents': {}
        }
        for node_id, agent in self.agents.items():
            checkpoint['agents'][node_id] = {
                'policy_net': agent.policy_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'neighbor_ids': agent.neighbor_ids,
                'num_neighbors': agent.num_neighbors
            }
        torch.save(checkpoint, path)
        print(f"[MultiAgent] Saved {len(self.agents)} agents to {path}")
    
    def load(self, path: str):
        """Load all agents' weights from a checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.global_steps = checkpoint['global_steps']
        
        for node_id, data in checkpoint['agents'].items():
            node_id = int(node_id)
            if node_id in self.agents:
                self.agents[node_id].policy_net.load_state_dict(data['policy_net'])
                self.agents[node_id].target_net.load_state_dict(data['target_net'])
                self.agents[node_id].optimizer.load_state_dict(data['optimizer'])
        
        print(f"[MultiAgent] Loaded {len(checkpoint['agents'])} agents from {path}")


if __name__ == "__main__":
    print("Testing MultiAgent components...")
    
    # Test NodeAgent creation
    agent = NodeAgent(
        node_id=0,
        num_nodes=50,
        num_neighbors=3,
        neighbor_ids=[1, 5, 12]
    )
    print(f"NodeAgent 0: {agent.num_neighbors} neighbors, "
          f"{sum(p.numel() for p in agent.policy_net.parameters())} params")
    
    # Test index conversion
    assert agent.neighbor_idx_to_node_id(0) == 1
    assert agent.neighbor_idx_to_node_id(2) == 12
    assert agent.node_id_to_neighbor_idx(5) == 1
    print("Index conversion: OK")
