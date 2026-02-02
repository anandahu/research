"""GCN model with Digital Twin awareness (7 input features)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import math
import random
from tqdm import tqdm
from itertools import count
from collections import namedtuple
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_dt_gcn.config import (
    device, NODE_FEATURE_DIM, EPS_START, EPS_END, EPS_DECAY,
    BATCH_SIZE, GAMMA, TARGET_UPDATE, LEARNING_RATE
)


# Transition for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class GCN_DTAware(nn.Module):
    """
    GCN model with Digital Twin awareness.
    
    Input: 7-dim node features (4 original + 3 DT-based)
    Architecture: GATConv -> GATConv -> Linear -> Linear
    Output: Q-values for each node (action = which neighbor to visit)
    """
    
    def __init__(self, num_nodes: int, out_dim: int, node_feat_dim: int = NODE_FEATURE_DIM):
        """
        Args:
            num_nodes: Number of nodes in graph (50)
            out_dim: Output dimension (50 for Q-value per node)
            node_feat_dim: Input feature dimension (7)
        """
        super().__init__()
        
        conv1_out = num_nodes // 3  # ~16
        conv2_out = num_nodes // 2  # ~25
        
        # Graph convolutions
        self.conv1 = GATConv(node_feat_dim, conv1_out, edge_dim=1)
        self.conv2 = GATConv(conv1_out, conv2_out, edge_dim=1)
        
        # Linear layers
        self.fc1 = nn.Linear(num_nodes * conv2_out, num_nodes)
        self.fc2 = nn.Linear(num_nodes, out_dim)
        
        self.num_nodes = num_nodes
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: Tuple of (DataLoader, mask_tensor)
                   DataLoader contains PyG Data objects
                   mask_tensor is valid action mask
                   
        Returns:
            Q-values for each node, masked for valid actions
        """
        loader, mask = state
        
        for batch in loader:
            batch = batch.to(device)
            out = self.conv1(batch.x, batch.edge_index)
            out = F.selu(out)
            out = self.conv2(out, batch.edge_index)
        
        # Reshape: (batch_size * num_nodes, features) -> (batch_size, num_nodes * features)
        out = torch.stack(out.split(self.num_nodes))
        out = out.flatten(start_dim=1)
        
        # Linear layers
        out = F.selu(out)
        out = self.fc1(out)
        out = F.selu(out)
        out = self.fc2(out)
        
        # Mask invalid actions
        out[~mask] = float('-inf')
        
        return out


class GCN_DTAgent:
    """
    DQN agent using GCN_DTAware model.
    
    Epsilon-greedy exploration with experience replay.
    """
    
    def __init__(self, num_nodes: int, policy_net: GCN_DTAware, 
                 target_net: GCN_DTAware, env):
        """
        Args:
            num_nodes: Number of nodes (50)
            policy_net: Policy GCN network
            target_net: Target GCN network (for stable Q-learning)
            env: HybridEnv environment
        """
        self.num_nodes = num_nodes
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.env = env
        self.memory = ReplayMemory(5000)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        self.steps_done = 0
        self.episode_durations = []
        
        # Metrics tracking
        self.metrics = {
            'loss': [],
            'reward': [],
            'path_length': [],
            'eps_reward': [],
            'jammed_steps': [],
            'success_rate': [],
            # New: per-step metrics
            'step_latency': [],
            'step_bandwidth': [],
            'step_pdr': [],
            # New: per-episode aggregates
            'eps_total_latency': [],
            'eps_avg_bandwidth': [],
            'eps_avg_pdr': []
        }
        
    def select_action(self, state, valid_actions):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: PyG Data object
            valid_actions: Boolean mask of valid actions
            
        Returns:
            Action tensor
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        
        if sample > eps_threshold:
            # Exploit: use policy network
            state_loader = DataLoader([state], batch_size=1, shuffle=False)
            with torch.no_grad():
                self.policy_net.eval()
                q_values = self.policy_net((state_loader, valid_actions))
                action = q_values.max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            # Explore: random valid action
            valid_indices = valid_actions[0].nonzero().squeeze()
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            random_idx = torch.randint(0, len(valid_indices), (1,))
            return valid_indices[random_idx].view(1, 1)
    
    def optimize_model(self):
        """Perform one optimization step."""
        if len(self.memory) < BATCH_SIZE:
            return
            
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Compute mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        
        # Prepare batches
        non_final_next_graphs = [s[0] for s in batch.next_state if s is not None]
        non_final_next_states = (
            DataLoader(non_final_next_graphs, batch_size=len(non_final_next_graphs), shuffle=False),
            torch.cat([s[1] for s in batch.next_state if s is not None])
        )
        
        state_graphs = [s[0] for s in batch.state]
        state_batch = (
            DataLoader(state_graphs, batch_size=len(state_graphs), shuffle=False),
            torch.cat([s[1] for s in batch.state])
        )
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s') for all next states
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if len(non_final_next_graphs) > 0:
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.metrics['loss'].append(loss.item())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def run(self, num_episodes: int = 1000, verbose: bool = True):
        """
        Run training loop.
        
        Args:
            num_episodes: Number of episodes to train
            verbose: Print progress
        """
        step_count = 0
        successes = 0
        
        iterator = tqdm(range(num_episodes), desc="Training") if verbose else range(num_episodes)
        
        for episode in iterator:
            # Reset environment
            obs, info = self.env.reset()
            valid_actions = info['valid_actions']
            state = (obs, valid_actions)
            
            eps_reward = 0
            jammed_count = 0
            eps_latency = 0
            eps_bandwidths = []
            eps_pdrs = []
            
            for t in count():
                step_count += 1
                self.steps_done += 1
                
                # Select and perform action
                action = self.select_action(obs, valid_actions)
                next_obs, reward, done, info = self.env.step(action.item())
                
                eps_reward += reward
                self.metrics['reward'].append(reward)
                
                # Track if stepped on jammed
                if action.item() in self.env.jammed_nodes:
                    jammed_count += 1
                
                # Track latency and bandwidth for this step
                prev_node = self.env.path[-2] if len(self.env.path) >= 2 else self.env.path[-1]
                curr_node = action.item()
                try:
                    step_latency = self.env.graph[prev_node][curr_node]['weight']
                    step_bandwidth = self.env.graph[prev_node][curr_node]['capacity']
                except:
                    step_latency = 0
                    step_bandwidth = 0
                
                self.metrics['step_latency'].append(step_latency)
                self.metrics['step_bandwidth'].append(step_bandwidth)
                # Use bandwidth/capacity as proxy for Link PDR
                self.metrics['step_pdr'].append(step_bandwidth)
                eps_latency += step_latency
                eps_bandwidths.append(step_bandwidth)
                eps_pdrs.append(step_bandwidth)
                
                next_valid = info['valid_actions']
                next_state = (next_obs, next_valid) if not done else None
                
                # Store transition
                reward_tensor = torch.tensor([reward], device=device)
                self.memory.push(state, action, next_state, reward_tensor)
                
                # Move to next state
                state = next_state
                obs = next_obs
                valid_actions = next_valid
                
                # Optimize
                self.optimize_model()
                
                if done:
                    self.episode_durations.append(t + 1)
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
                    
                    if reward > 0:
                        successes += 1
                    break
                
                # Update target network
                if step_count % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Progress logging
            if verbose and episode % 50 == 0 and episode > 0:
                recent_rewards = self.metrics['eps_reward'][-50:]
                success_rate = successes / (episode + 1)
                print(f"\nEpisode {episode}: avg_reward={np.mean(recent_rewards):.2f}, "
                      f"success_rate={success_rate:.2%}")
                
        # Final stats
        if verbose:
            print(f"\nTraining complete. Final success rate: {successes/num_episodes:.2%}")
            
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']


# Need numpy for metrics
import numpy as np


if __name__ == "__main__":
    print("Testing GCN_DTAware model...")
    
    model = GCN_DTAware(num_nodes=50, out_dim=50, node_feat_dim=7)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    from torch_geometric.data import Data
    x = torch.randn(50, 7)
    edge_index = torch.randint(0, 50, (2, 100))
    data = Data(x=x, edge_index=edge_index)
    
    loader = DataLoader([data], batch_size=1)
    mask = torch.ones((1, 50), dtype=torch.bool)
    
    out = model((loader, mask))
    print(f"Output shape: {out.shape}")
