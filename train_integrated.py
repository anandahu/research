"""
Main training script for Digital Twin + GCN integrated system.

Usage:
    python train_integrated.py --episodes 1000
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_dt_gcn.config import device, NUM_NODES, NODE_FEATURE_DIM
from integrated_dt_gcn.brite_loader import BRITELoader
from integrated_dt_gcn.dataset_loader import DTDatasetLoader
from integrated_dt_gcn.digital_twin.anomaly_bridge import AnomalyBridge
from integrated_dt_gcn.environment.hybrid_env import HybridEnv
from integrated_dt_gcn.models.gcn_dt_aware import GCN_DTAware, GCN_DTAgent


def main(args):
    """Main training function."""
    
    print("=" * 60)
    print("Digital Twin + GCN Integrated Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Dataset: {args.dataset_nr}")
    print()
    
    # Paths
    brite_path = os.path.join(os.path.dirname(__file__), "RP15", "50nodes.brite")
    dataset_dir = os.path.join(os.path.dirname(__file__), "RP12_paper", "datasets")
    
    # 1. Load DT datasets
    print("Loading DT datasets...")
    loader = DTDatasetLoader(dataset_dir)
    
    num_scenarios = loader.get_scenario_count(args.dataset_nr)
    num_jammed = len(loader.get_jammed_indices(args.dataset_nr))
    num_normal = len(loader.get_normal_indices(args.dataset_nr))
    print(f"  Total scenarios: {num_scenarios}")
    print(f"  Jammed: {num_jammed}, Normal: {num_normal}")
    
    # 2. Train anomaly detector on normal samples
    print("\nTraining anomaly detector...")
    anomaly_bridge = AnomalyBridge()
    normal_measurements = loader.get_normal_measurements(args.dataset_nr)[:args.normal_samples]
    anomaly_bridge.train(normal_measurements)
    
    # 3. Create hybrid environment
    print("\nCreating hybrid environment...")
    env = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
    print(f"  Nodes: {env.num_nodes}")
    print(f"  Node features: {NODE_FEATURE_DIM}")
    
    # 4. Create GCN model
    print("\nCreating GCN_DTAware model...")
    policy_net = GCN_DTAware(
        num_nodes=env.num_nodes,
        out_dim=env.num_nodes,
        node_feat_dim=NODE_FEATURE_DIM
    )
    target_net = GCN_DTAware(
        num_nodes=env.num_nodes,
        out_dim=env.num_nodes,
        node_feat_dim=NODE_FEATURE_DIM
    )
    
    num_params = sum(p.numel() for p in policy_net.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # 5. Create agent
    agent = GCN_DTAgent(
        num_nodes=env.num_nodes,
        policy_net=policy_net,
        target_net=target_net,
        env=env
    )
    
    # 6. Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    agent.run(num_episodes=args.episodes, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # 7. Save metrics to CSV (always)
    save_metrics_to_csv(agent.metrics, args.episodes, elapsed)
    
    # 8. Save model
    if args.save:
        save_path = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "trained_model.pt")
        agent.save(save_path)
        print(f"Model saved to: {save_path}")
    
    # 9. Plot metrics
    if args.plot:
        plot_metrics(agent.metrics)
    
    return agent


def save_metrics_to_csv(metrics: dict, num_episodes: int, elapsed_time: float):
    """Save all training metrics to CSV files."""
    
    output_dir = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Episode-level metrics CSV
    episode_df = pd.DataFrame({
        'episode': range(1, len(metrics['eps_reward']) + 1),
        'episode_reward': metrics['eps_reward'],
        'path_length': metrics['path_length'],
        'jammed_steps': metrics['jammed_steps'],
        'total_latency': metrics.get('eps_total_latency', [0] * len(metrics['eps_reward'])),
        'avg_bandwidth': metrics.get('eps_avg_bandwidth', [0] * len(metrics['eps_reward'])),
        'avg_pdr': metrics.get('eps_avg_pdr', [0] * len(metrics['eps_reward']))
    })
    episode_path = os.path.join(output_dir, f"episode_metrics_{timestamp}.csv")
    episode_df.to_csv(episode_path, index=False)
    print(f"Episode metrics saved to: {episode_path}")
    
    # 2. Step-level metrics CSV (loss, latency, bandwidth)
    if metrics['loss']:
        step_df = pd.DataFrame({
            'step': range(1, len(metrics['loss']) + 1),
            'loss': metrics['loss']
        })
        # Add step latency/bandwidth if available
        if metrics.get('step_latency'):
            step_df['latency'] = metrics['step_latency'][:len(metrics['loss'])]
        if metrics.get('step_bandwidth'):
            step_df['bandwidth'] = metrics['step_bandwidth'][:len(metrics['loss'])]
        
        step_path = os.path.join(output_dir, f"step_metrics_{timestamp}.csv")
        step_df.to_csv(step_path, index=False)
        print(f"Step metrics saved to: {step_path}")
    
    # 3. Summary CSV (append to running file for comparison)
    successes = sum(1 for r in metrics['eps_reward'] if r > 0)
    
    # Calculate latency/bandwidth/PDR averages
    avg_latency = np.mean(metrics.get('eps_total_latency', [0])) if metrics.get('eps_total_latency') else 0
    avg_bandwidth = np.mean(metrics.get('eps_avg_bandwidth', [0])) if metrics.get('eps_avg_bandwidth') else 0
    avg_pdr = np.mean(metrics.get('eps_avg_pdr', [0])) if metrics.get('eps_avg_pdr') else 0
    
    summary = {
        'timestamp': timestamp,
        'model': 'GCN_DTAware',
        'episodes': num_episodes,
        'training_time_min': round(elapsed_time / 60, 2),
        'avg_reward': round(np.mean(metrics['eps_reward']), 3),
        'max_reward': round(max(metrics['eps_reward']), 3),
        'min_reward': round(min(metrics['eps_reward']), 3),
        'success_rate': round(successes / len(metrics['eps_reward']), 4),
        'avg_path_length': round(np.mean(metrics['path_length']), 2),
        'avg_jammed_steps': round(np.mean(metrics['jammed_steps']), 2),
        'avg_latency': round(avg_latency, 4),
        'avg_bandwidth': round(avg_bandwidth, 4),
        'avg_pdr': round(avg_pdr, 4),
        'final_50_avg_reward': round(np.mean(metrics['eps_reward'][-50:]), 3) if len(metrics['eps_reward']) >= 50 else None
    }
    
    summary_path = os.path.join(output_dir, "training_summary.csv")
    summary_df = pd.DataFrame([summary])
    
    # Append if exists, else create
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    print(f"Summary appended to: {summary_path}")
    
    # 4. Print summary table
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def plot_metrics(metrics: dict):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    if metrics['eps_reward']:
        axes[0, 0].plot(metrics['eps_reward'])
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        # Moving average
        if len(metrics['eps_reward']) > 20:
            ma = np.convolve(metrics['eps_reward'], np.ones(20)/20, mode='valid')
            axes[0, 0].plot(range(19, len(metrics['eps_reward'])), ma, 'r-', label='MA(20)')
            axes[0, 0].legend()
    
    # Loss
    if metrics['loss']:
        axes[0, 1].plot(metrics['loss'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
    
    # Path length
    if metrics['path_length']:
        axes[1, 0].plot(metrics['path_length'])
        axes[1, 0].set_title('Path Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
    
    # Jammed steps
    if metrics['jammed_steps']:
        axes[1, 1].plot(metrics['jammed_steps'])
        axes[1, 1].set_title('Jammed Steps per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "training_metrics.png")
    plt.savefig(save_path)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DT + GCN integrated model")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--dataset_nr", type=int, default=0, help="Dataset number (0-5)")
    parser.add_argument("--normal_samples", type=int, default=2000, help="Normal samples for anomaly training")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    parser.add_argument("--plot", action="store_true", help="Plot training metrics")
    
    args = parser.parse_args()
    main(args)
