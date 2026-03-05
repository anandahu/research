"""
Multi-Agent training script for Digital Twin + GCN integrated system.

Uses RP15-style multi-agent: one GCN agent per node (50 agents total).

Usage:
    python train_multi_agent.py --episodes 1000
    python train_multi_agent.py --episodes 500 --save --plot
"""

import argparse
import os
import sys
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_dt_gcn.config import device, NUM_NODES, NODE_FEATURE_DIM
from integrated_dt_gcn.dataset_loader import DTDatasetLoader
from integrated_dt_gcn.digital_twin.anomaly_bridge import AnomalyBridge
from integrated_dt_gcn.environment.hybrid_env import HybridEnv
from integrated_dt_gcn.models.multi_agent import MultiAgentCoordinator


def main(args):
    """Main training function for multi-agent system."""
    
    # Set random seeds for reproducibility (Fix Gap 2)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    print("=" * 60)
    print("Digital Twin + GAT Multi-Agent Training (RP15-style)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Dataset: {args.dataset_nr}")
    print(f"Random seed: {SEED}")
    print(f"Architecture: 50 independent GAT agents (1 per node)")
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
    
    # 4. Create Multi-Agent Coordinator
    print("\nCreating Multi-Agent Coordinator...")
    coordinator = MultiAgentCoordinator(env)
    
    # 5. Train
    print("\n" + "=" * 60)
    print("Starting multi-agent training...")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    coordinator.run(num_episodes=args.episodes, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # 6. Test
    if args.test_episodes > 0:
        print(f"\nRunning {args.test_episodes} test episodes...")
        success_rate = coordinator.test(num_episodes=args.test_episodes)
    
    # 7. Save metrics to CSV
    save_metrics_to_csv(coordinator.metrics, args.episodes, elapsed)
    
    # 8. Save model
    if args.save:
        save_path = os.path.join(os.path.dirname(__file__), 
                                  "integrated_dt_gcn", "trained_multi_agent.pt")
        coordinator.save(save_path)
        print(f"Model saved to: {save_path}")
    
    # 9. Plot metrics
    if args.plot:
        plot_metrics(coordinator.metrics)
    
    return coordinator


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
    episode_path = os.path.join(output_dir, f"ma_episode_metrics_{timestamp}.csv")
    episode_df.to_csv(episode_path, index=False)
    print(f"Episode metrics saved to: {episode_path}")
    
    # 2. Summary CSV
    successes = sum(1 for r in metrics['eps_reward'] if r > 0)
    
    avg_latency = np.mean(metrics.get('eps_total_latency', [0])) if metrics.get('eps_total_latency') else 0
    avg_bandwidth = np.mean(metrics.get('eps_avg_bandwidth', [0])) if metrics.get('eps_avg_bandwidth') else 0
    avg_pdr = np.mean(metrics.get('eps_avg_pdr', [0])) if metrics.get('eps_avg_pdr') else 0
    
    summary = {
        'timestamp': timestamp,
        'model': 'MultiAgent_GAT_DTAware',
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
    
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    print(f"Summary appended to: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("MULTI-AGENT TRAINING SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def plot_metrics(metrics: dict):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Multi-Agent GAT+DT Training Metrics', fontsize=14)
    
    # Episode rewards
    if metrics['eps_reward']:
        axes[0, 0].plot(metrics['eps_reward'], alpha=0.3, color='blue')
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        if len(metrics['eps_reward']) > 20:
            ma = np.convolve(metrics['eps_reward'], np.ones(20)/20, mode='valid')
            axes[0, 0].plot(range(19, len(metrics['eps_reward'])), ma, 'r-', 
                          label='MA(20)', linewidth=2)
            axes[0, 0].legend()
    
    # Loss
    if metrics['loss']:
        axes[0, 1].plot(metrics['loss'], alpha=0.3, color='orange')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
    
    # Path length
    if metrics['path_length']:
        axes[1, 0].plot(metrics['path_length'], alpha=0.3, color='green')
        axes[1, 0].set_title('Path Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        if len(metrics['path_length']) > 20:
            ma = np.convolve(metrics['path_length'], np.ones(20)/20, mode='valid')
            axes[1, 0].plot(range(19, len(metrics['path_length'])), ma, 'r-', linewidth=2)
    
    # Jammed steps
    if metrics['jammed_steps']:
        axes[1, 1].plot(metrics['jammed_steps'], alpha=0.3, color='red')
        axes[1, 1].set_title('Jammed Steps per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 
                              "integrated_dt_gcn", "ma_training_metrics.png")
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Agent DT + GAT model")
    parser.add_argument("--episodes", type=int, default=500, 
                       help="Number of training episodes")
    parser.add_argument("--dataset_nr", type=int, default=0, 
                       help="Dataset number (0-5)")
    parser.add_argument("--normal_samples", type=int, default=2000, 
                       help="Normal samples for anomaly training")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--test_episodes", type=int, default=200, 
                       help="Number of test episodes (0 to skip)")
    parser.add_argument("--save", action="store_true", 
                       help="Save trained models")
    parser.add_argument("--plot", action="store_true", 
                       help="Plot training metrics")
    
    args = parser.parse_args()
    main(args)
