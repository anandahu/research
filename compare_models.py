"""
Compare Multi-Agent GCN+DT vs Single-Agent GCN Baseline.

Usage:
    python compare_models.py --episodes 500
    python compare_models.py --episodes 1000 --plot
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_dt_gcn.config import device, NUM_NODES, NODE_FEATURE_DIM
from integrated_dt_gcn.brite_loader import BRITELoader
from integrated_dt_gcn.dataset_loader import DTDatasetLoader
from integrated_dt_gcn.digital_twin.anomaly_bridge import AnomalyBridge
from integrated_dt_gcn.environment.hybrid_env import HybridEnv
from integrated_dt_gcn.models.gat_dt_aware import GAT_DTAware, GAT_DTAgent
from integrated_dt_gcn.models.multi_agent import MultiAgentCoordinator


def train_single_agent(env, num_nodes: int, node_feat_dim: int,
                       num_episodes: int, name: str) -> dict:
    """Train single-agent GCN model and return metrics."""
    
    print(f"\nTraining {name}...")
    print("-" * 40)
    
    policy_net = GAT_DTAware(num_nodes, num_nodes, node_feat_dim)
    target_net = GAT_DTAware(num_nodes, num_nodes, node_feat_dim)
    
    agent = GAT_DTAgent(num_nodes, policy_net, target_net, env)
    
    start_time = time.time()
    agent.run(num_episodes, verbose=True)
    elapsed = time.time() - start_time
    
    # Calculate final metrics (last 100 episodes)
    final_rewards = agent.metrics['eps_reward'][-100:] if len(agent.metrics['eps_reward']) >= 100 else agent.metrics['eps_reward']
    final_path_lengths = agent.metrics['path_length'][-100:] if len(agent.metrics['path_length']) >= 100 else agent.metrics['path_length']
    final_jammed = agent.metrics['jammed_steps'][-100:] if len(agent.metrics['jammed_steps']) >= 100 else agent.metrics['jammed_steps']
    
    # Use actual target-reached tracking instead of reward > 0
    final_success = agent.metrics['episode_success'][-100:] if len(agent.metrics['episode_success']) >= 100 else agent.metrics['episode_success']
    successes = sum(final_success)
    
    return {
        'name': name,
        'time': elapsed,
        'avg_reward': np.mean(final_rewards),
        'avg_path_length': np.mean(final_path_lengths),
        'avg_jammed_steps': np.mean(final_jammed),
        'success_rate': successes / len(final_success),
        'all_rewards': agent.metrics['eps_reward'],
        'all_losses': agent.metrics['loss'],
        'avg_pdr': np.mean(agent.metrics.get('eps_avg_pdr', [0])),
        'all_pdrs': agent.metrics.get('eps_avg_pdr', []),
        'all_path_lengths': agent.metrics['path_length'],
        'all_jammed_steps': agent.metrics['jammed_steps'],
        'all_latency': agent.metrics.get('eps_total_latency', []),
        'all_success': agent.metrics.get('episode_success', [])
    }


def train_multi_agent(env, num_episodes: int, name: str) -> dict:
    """Train multi-agent GCN+DT model and return metrics."""
    
    print(f"\nTraining {name}...")
    print("-" * 40)
    
    coordinator = MultiAgentCoordinator(env)
    
    start_time = time.time()
    coordinator.run(num_episodes, verbose=True)
    elapsed = time.time() - start_time
    
    # Calculate final metrics (last 100 episodes)
    final_rewards = coordinator.metrics['eps_reward'][-100:] if len(coordinator.metrics['eps_reward']) >= 100 else coordinator.metrics['eps_reward']
    final_path_lengths = coordinator.metrics['path_length'][-100:] if len(coordinator.metrics['path_length']) >= 100 else coordinator.metrics['path_length']
    final_jammed = coordinator.metrics['jammed_steps'][-100:] if len(coordinator.metrics['jammed_steps']) >= 100 else coordinator.metrics['jammed_steps']
    
    # Use actual target-reached tracking instead of reward > 0
    final_success = coordinator.metrics['episode_success'][-100:] if len(coordinator.metrics['episode_success']) >= 100 else coordinator.metrics['episode_success']
    successes = sum(final_success)
    
    return {
        'name': name,
        'time': elapsed,
        'avg_reward': np.mean(final_rewards),
        'avg_path_length': np.mean(final_path_lengths),
        'avg_jammed_steps': np.mean(final_jammed),
        'success_rate': successes / len(final_success),
        'all_rewards': coordinator.metrics['eps_reward'],
        'all_losses': coordinator.metrics['loss'],
        'avg_pdr': np.mean(coordinator.metrics.get('eps_avg_pdr', [0])),
        'all_pdrs': coordinator.metrics.get('eps_avg_pdr', []),
        'all_path_lengths': coordinator.metrics['path_length'],
        'all_jammed_steps': coordinator.metrics['jammed_steps'],
        'all_latency': coordinator.metrics.get('eps_total_latency', []),
        'all_success': coordinator.metrics.get('episode_success', [])
    }


def main(args):
    """Run comparison experiments."""
    
    # Set random seeds for reproducibility (Fix Gap 2)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    print("=" * 60)
    print("Multi-Agent GAT+DT vs GAT Baseline Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Random seed: {SEED}")
    
    # Paths
    brite_path = os.path.join(os.path.dirname(__file__), "RP15", "50nodes.brite")
    dataset_dir = os.path.join(os.path.dirname(__file__), "RP12_paper", "datasets")
    
    # Load data
    print("\nLoading datasets...")
    loader = DTDatasetLoader(dataset_dir)
    
    # Train anomaly detector
    print("Training anomaly detector...")
    anomaly_bridge = AnomalyBridge()
    normal_measurements = loader.get_normal_measurements(args.dataset_nr)[:2000]
    anomaly_bridge.train(normal_measurements)
    
    results = []
    
    # 1. Multi-Agent GCN + DT (50 agents, 7 features)
    env_ma = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
    result_ma = train_multi_agent(env_ma, args.episodes, "Multi-Agent GAT+DT (50 agents)")
    results.append(result_ma)
    
    # 2. Single-Agent GCN Baseline (4 features - DT zeroed out)
    env_baseline = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
    original_compute = env_baseline._compute_node_features
    def compute_4_features(self=env_baseline):
        x = original_compute()
        x[:, 4:] = 0  # Zero out DT features
        return x
    env_baseline._compute_node_features = compute_4_features
    
    result_baseline = train_single_agent(env_baseline, NUM_NODES, NODE_FEATURE_DIM, 
                                          args.episodes, "GAT Baseline (4 features)")
    results.append(result_baseline)
    
    # Print comparison table
    print("\n" + "=" * 65)
    print("COMPARISON RESULTS")
    print("=" * 65)
    print(f"{'Metric':<25} {'MA GAT+DT':<20} {'GAT Baseline':<20}")
    print("-" * 65)
    print(f"{'Avg Reward':<25} {result_ma['avg_reward']:<20.3f} {result_baseline['avg_reward']:<20.3f}")
    print(f"{'Success Rate':<25} {result_ma['success_rate']:<20.2%} {result_baseline['success_rate']:<20.2%}")
    print(f"{'Avg Path Length':<25} {result_ma['avg_path_length']:<20.2f} {result_baseline['avg_path_length']:<20.2f}")
    print(f"{'Avg Jammed Steps':<25} {result_ma['avg_jammed_steps']:<20.2f} {result_baseline['avg_jammed_steps']:<20.2f}")
    print(f"{'Training Time (min)':<25} {result_ma['time']/60:<20.1f} {result_baseline['time']/60:<20.1f}")
    print("=" * 65)
    
    # Save comparison to CSV
    save_comparison_to_csv(results, args.episodes)
    
    # Plot comparison
    if args.plot:
        plot_comparison(results)
    
    return results


def save_comparison_to_csv(results: list, num_episodes: int):
    """Save comparison results to CSV files."""
    
    output_dir = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Per-episode comparison CSV (side by side)
    max_eps = max(len(r['all_rewards']) for r in results)
    episode_data = {'episode': range(1, max_eps + 1)}
    
    for r in results:
        name = r['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        rewards = r['all_rewards'] + [None] * (max_eps - len(r['all_rewards']))
        episode_data[f'{name}_reward'] = rewards
        
        path_lengths = r.get('all_path_lengths', []) + [None] * (max_eps - len(r.get('all_path_lengths', [])))
        episode_data[f'{name}_path_length'] = path_lengths
        
        jammed = r.get('all_jammed_steps', []) + [None] * (max_eps - len(r.get('all_jammed_steps', [])))
        episode_data[f'{name}_jammed_steps'] = jammed
        
        latency = r.get('all_latency', []) + [None] * (max_eps - len(r.get('all_latency', [])))
        episode_data[f'{name}_latency'] = latency
    
    episode_df = pd.DataFrame(episode_data)
    episode_path = os.path.join(output_dir, f"comparison_episodes_{timestamp}.csv")
    episode_df.to_csv(episode_path, index=False)
    print(f"\nComparison episodes saved to: {episode_path}")
    
    # 2. Summary comparison CSV
    summary_data = []
    for r in results:
        avg_pdr = np.mean(r.get('all_pdrs', [0])) if r.get('all_pdrs') else 0
        
        summary_data.append({
            'timestamp': timestamp,
            'model': r['name'],
            'episodes': num_episodes,
            'training_time_min': round(r['time'] / 60, 2),
            'avg_reward': round(r['avg_reward'], 3),
            'success_rate': round(r['success_rate'], 4),
            'avg_path_length': round(r['avg_path_length'], 2),
            'avg_jammed_steps': round(r['avg_jammed_steps'], 2),
            'avg_pdr': round(avg_pdr, 4)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"comparison_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Comparison summary saved to: {summary_path}")
    
    # 3. Improvement metrics
    if len(results) >= 2:
        improvement = {
            'timestamp': timestamp,
            'ma_model': results[0]['name'],
            'baseline_model': results[1]['name'],
            'reward_improvement': round(results[0]['avg_reward'] - results[1]['avg_reward'], 3),
            'success_rate_improvement_pp': round((results[0]['success_rate'] - results[1]['success_rate']) * 100, 2),
            'jammed_steps_reduction': round(results[1]['avg_jammed_steps'] - results[0]['avg_jammed_steps'], 2),
            'path_length_reduction': round(results[1]['avg_path_length'] - results[0]['avg_path_length'], 2),
            'pdr_improvement': round(results[0].get('avg_pdr', 0) - results[1].get('avg_pdr', 0), 4)
        }
        
        improvement_path = os.path.join(output_dir, "comparison_improvements.csv")
        improvement_df = pd.DataFrame([improvement])
        
        if os.path.exists(improvement_path):
            improvement_df.to_csv(improvement_path, mode='a', header=False, index=False)
        else:
            improvement_df.to_csv(improvement_path, index=False)
        print(f"Improvement metrics appended to: {improvement_path}")


def plot_comparison(results: list):
    """Plot comparison charts."""
    
    output_dir = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "results")
    colors = ['#2ecc71', '#e74c3c']
    labels = ['MA GAT+DT', 'GAT Baseline']
    
    # --- 1. Reward convergence line graph ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        if r['all_rewards']:
            rewards_series = pd.Series(r['all_rewards'])
            ax.plot(rewards_series.rolling(50).mean(), 
                    label=labels[i], color=colors[i], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (50-episode moving avg)', fontsize=12)
    ax.set_title('Multi-Agent GAT+DT vs GAT Baseline: Reward Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_reward_graph.png"), dpi=150, bbox_inches='tight')
    print("Reward comparison graph saved.")
    plt.close()
    
    # --- 2. Path length convergence line graph ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        if r.get('all_path_lengths'):
            path_series = pd.Series(r['all_path_lengths'])
            ax.plot(path_series.rolling(50).mean(), 
                    label=labels[i], color=colors[i], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Path Length (50-episode moving avg)', fontsize=12)
    ax.set_title('Path Length Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_path_length_graph.png"), dpi=150, bbox_inches='tight')
    print("Path length comparison graph saved.")
    plt.close()
    
    # --- 3. Jammed steps convergence line graph ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        if r.get('all_jammed_steps'):
            jammed_series = pd.Series(r['all_jammed_steps'])
            ax.plot(jammed_series.rolling(50).mean(), 
                    label=labels[i], color=colors[i], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Jammed Steps (50-episode moving avg)', fontsize=12)
    ax.set_title('Jammed Steps Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_jammed_steps_graph.png"), dpi=150, bbox_inches='tight')
    print("Jammed steps comparison graph saved.")
    plt.close()
    
    # --- 4. Training time bar chart ---
    fig, ax = plt.subplots(figsize=(8, 6))
    times = [r['time'] / 60 for r in results]
    bars = ax.bar(labels, times, color=colors, edgecolor='black', linewidth=1.2)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{t:.1f} min', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title(f'Training Time Comparison ({len(results[0]["all_rewards"])} Episodes)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_time_graph.png"), dpi=150, bbox_inches='tight')
    print("Time comparison graph saved.")
    plt.close()
    
    # --- 5. PDR & Success Rate bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = [r['success_rate'] * 100 for r in results]
    pdr_values = [r.get('avg_pdr', 0) * 100 for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, pdr_values, width, label='PDR (%)', 
                   color='#9b59b6', edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars1, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, pdr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Success Rate & PDR Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_pdr_success_graph.png"), dpi=150, bbox_inches='tight')
    print("PDR/Success comparison graph saved.")
    plt.close()
    
    # --- 6. Combined 2x2 summary ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward curves
    for i, r in enumerate(results):
        if r['all_rewards']:
            rewards_series = pd.Series(r['all_rewards'])
            axes[0, 0].plot(rewards_series.rolling(50).mean(), label=labels[i], color=colors[i], linewidth=2)
    axes[0, 0].set_title('Reward Convergence', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Bar chart comparison
    metrics_names = ['Success Rate', 'Avg Reward / 2', 'Jammed Steps / 5']
    x = np.arange(len(metrics_names))
    width = 0.35
    values_ma = [results[0]['success_rate'], results[0]['avg_reward'] / 2, results[0]['avg_jammed_steps'] / 5]
    values_bl = [results[1]['success_rate'], results[1]['avg_reward'] / 2, results[1]['avg_jammed_steps'] / 5]
    axes[0, 1].bar(x - width/2, values_ma, width, label=labels[0], color=colors[0])
    axes[0, 1].bar(x + width/2, values_bl, width, label=labels[1], color=colors[1])
    axes[0, 1].set_title('Performance Comparison', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Loss curves
    for i, r in enumerate(results):
        if r['all_losses']:
            window = 100
            if len(r['all_losses']) > window:
                ma = np.convolve(r['all_losses'], np.ones(window)/window, mode='valid')
                axes[1, 0].plot(ma, label=labels[i], color=colors[i])
    axes[1, 0].set_title('Training Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary = f"""
    COMPARISON SUMMARY
    ==================
    
    Multi-Agent GAT+DT (50 agents):
      • Success Rate: {results[0]['success_rate']:.1%}
      • Avg Reward: {results[0]['avg_reward']:.3f}
      • Avg Path Length: {results[0]['avg_path_length']:.2f}
      • Avg Jammed Steps: {results[0]['avg_jammed_steps']:.2f}
      • Training Time: {results[0]['time']/60:.1f} min
    
    GAT Baseline (single-agent):
      • Success Rate: {results[1]['success_rate']:.1%}
      • Avg Reward: {results[1]['avg_reward']:.3f}
      • Avg Path Length: {results[1]['avg_path_length']:.2f}
      • Avg Jammed Steps: {results[1]['avg_jammed_steps']:.2f}
      • Training Time: {results[1]['time']/60:.1f} min
    
    Improvement (MA over Baseline):
      • Success: {(results[0]['success_rate'] - results[1]['success_rate'])*100:+.1f} pp
      • Jammed: {(results[1]['avg_jammed_steps'] - results[0]['avg_jammed_steps']):+.2f} steps avoided
    """
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Multi-Agent GAT+DT vs GAT Baseline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_results.png"), dpi=150, bbox_inches='tight')
    print("Combined comparison plot saved.")
    plt.close()
    
    print(f"\nAll comparison graphs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Multi-Agent GAT+DT vs GAT Baseline")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per model")
    parser.add_argument("--dataset_nr", type=int, default=0, help="Dataset number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    
    args = parser.parse_args()
    main(args)
