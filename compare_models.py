"""
Compare GCN with and without Digital Twin integration.

Usage:
    python compare_models.py --episodes 500
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_dt_gcn.config import device, NUM_NODES, NODE_FEATURE_DIM
from integrated_dt_gcn.brite_loader import BRITELoader
from integrated_dt_gcn.dataset_loader import DTDatasetLoader
from integrated_dt_gcn.digital_twin.anomaly_bridge import AnomalyBridge
from integrated_dt_gcn.environment.hybrid_env import HybridEnv
from integrated_dt_gcn.models.gcn_dt_aware import GCN_DTAware, GCN_DTAgent


def create_baseline_env(brite_path: str, loader: DTDatasetLoader, 
                        anomaly_bridge: AnomalyBridge, dataset_nr: int):
    """Create baseline environment (GCN without DT features)."""
    # Use same environment but we'll modify feature computation
    return HybridEnv(brite_path, loader, anomaly_bridge, dataset_nr)


def train_model(env, num_nodes: int, node_feat_dim: int, 
                num_episodes: int, name: str) -> dict:
    """Train a model and return metrics."""
    
    print(f"\nTraining {name}...")
    print("-" * 40)
    
    policy_net = GCN_DTAware(num_nodes, num_nodes, node_feat_dim)
    target_net = GCN_DTAware(num_nodes, num_nodes, node_feat_dim)
    
    agent = GCN_DTAgent(num_nodes, policy_net, target_net, env)
    
    start_time = time.time()
    agent.run(num_episodes, verbose=True)
    elapsed = time.time() - start_time
    
    # Calculate final metrics
    final_rewards = agent.metrics['eps_reward'][-100:] if len(agent.metrics['eps_reward']) >= 100 else agent.metrics['eps_reward']
    final_path_lengths = agent.metrics['path_length'][-100:] if len(agent.metrics['path_length']) >= 100 else agent.metrics['path_length']
    final_jammed = agent.metrics['jammed_steps'][-100:] if len(agent.metrics['jammed_steps']) >= 100 else agent.metrics['jammed_steps']
    
    successes = sum(1 for r in final_rewards if r > 0)
    
    return {
        'name': name,
        'time': elapsed,
        'avg_reward': np.mean(final_rewards),
        'avg_path_length': np.mean(final_path_lengths),
        'avg_jammed_steps': np.mean(final_jammed),
        'success_rate': successes / len(final_rewards),
        'all_rewards': agent.metrics['eps_reward'],
        'all_losses': agent.metrics['loss'],
        'avg_pdr': np.mean(agent.metrics.get('eps_avg_pdr', [0])),
        'all_pdrs': agent.metrics.get('eps_avg_pdr', [])
    }


def main(args):
    """Run comparison experiments."""
    
    print("=" * 60)
    print("GCN vs GCN+DT Comparison")
    print("=" * 60)
    
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
    
    # 1. GCN with DT (7 features)
    env_dt = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
    result_dt = train_model(env_dt, NUM_NODES, NODE_FEATURE_DIM, args.episodes, "GCN + DT (7 features)")
    results.append(result_dt)
    
    # 2. GCN without DT (4 features) - we use same env but baseline features
    # For fair comparison, create separate environment instance
    env_baseline = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
    # Monkey-patch to use only 4 features (disable DT features)
    original_compute = env_baseline._compute_node_features
    def compute_4_features(self=env_baseline):
        x = original_compute()
        x[:, 4:] = 0  # Zero out DT features
        return x
    env_baseline._compute_node_features = compute_4_features
    
    result_baseline = train_model(env_baseline, NUM_NODES, NODE_FEATURE_DIM, args.episodes, "GCN Baseline (4 features)")
    results.append(result_baseline)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'GCN + DT':<15} {'GCN Baseline':<15}")
    print("-" * 55)
    print(f"{'Avg Reward':<25} {result_dt['avg_reward']:<15.3f} {result_baseline['avg_reward']:<15.3f}")
    print(f"{'Success Rate':<25} {result_dt['success_rate']:<15.2%} {result_baseline['success_rate']:<15.2%}")
    print(f"{'Avg Path Length':<25} {result_dt['avg_path_length']:<15.2f} {result_baseline['avg_path_length']:<15.2f}")
    print(f"{'Avg Jammed Steps':<25} {result_dt['avg_jammed_steps']:<15.2f} {result_baseline['avg_jammed_steps']:<15.2f}")
    print(f"{'Training Time (min)':<25} {result_dt['time']/60:<15.1f} {result_baseline['time']/60:<15.1f}")
    print("=" * 60)
    
    # Save comparison to CSV (always)
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
        name = r['name'].replace(' ', '_').replace('(', '').replace(')', '')
        rewards = r['all_rewards'] + [None] * (max_eps - len(r['all_rewards']))
        episode_data[f'{name}_reward'] = rewards
    
    episode_df = pd.DataFrame(episode_data)
    episode_path = os.path.join(output_dir, f"comparison_episodes_{timestamp}.csv")
    episode_df.to_csv(episode_path, index=False)
    print(f"\nComparison episodes saved to: {episode_path}")
    
    # 2. Summary comparison CSV
    summary_data = []
    for r in results:
        # Calculate PDR if available
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
            'dt_model': results[0]['name'],
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#2ecc71', '#3498db']
    
    # 1. Reward curves
    for i, r in enumerate(results):
        if r['all_rewards']:
            # Moving average
            window = 20
            if len(r['all_rewards']) > window:
                ma = np.convolve(r['all_rewards'], np.ones(window)/window, mode='valid')
                axes[0, 0].plot(ma, label=r['name'], color=colors[i])
    axes[0, 0].set_title('Episode Reward (Moving Average)', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bar chart comparison
    metrics = ['Success Rate', 'Avg Reward', 'Jammed Steps']
    x = np.arange(len(metrics))
    width = 0.35
    
    values_dt = [results[0]['success_rate'], results[0]['avg_reward'] / 2, results[0]['avg_jammed_steps'] / 5]
    values_baseline = [results[1]['success_rate'], results[1]['avg_reward'] / 2, results[1]['avg_jammed_steps'] / 5]
    
    rects1 = axes[0, 1].bar(x - width/2, values_dt, width, label='GCN + DT', color=colors[0])
    rects2 = axes[0, 1].bar(x + width/2, values_baseline, width, label='GCN Baseline', color=colors[1])
    axes[0, 1].set_title('Performance Comparison', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Loss curves
    for i, r in enumerate(results):
        if r['all_losses']:
            window = 100
            if len(r['all_losses']) > window:
                ma = np.convolve(r['all_losses'], np.ones(window)/window, mode='valid')
                axes[1, 0].plot(ma, label=r['name'], color=colors[i])
    axes[1, 0].set_title('Training Loss', fontsize=12)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary text
    axes[1, 1].axis('off')
    summary = f"""
    COMPARISON SUMMARY
    ==================
    
    GCN + Digital Twin:
      • Success Rate: {results[0]['success_rate']:.1%}
      • Avg Reward: {results[0]['avg_reward']:.3f}
      • Avg Jammed Steps: {results[0]['avg_jammed_steps']:.2f}
    
    GCN Baseline:
      • Success Rate: {results[1]['success_rate']:.1%}
      • Avg Reward: {results[1]['avg_reward']:.3f}
      • Avg Jammed Steps: {results[1]['avg_jammed_steps']:.2f}
    
    Improvement with DT:
      • Success: {(results[0]['success_rate'] - results[1]['success_rate'])*100:+.1f} pp
      • Jammed: {(results[1]['avg_jammed_steps'] - results[0]['avg_jammed_steps']):+.2f} steps avoided
    """
    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "integrated_dt_gcn", "comparison_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GCN vs GCN+DT")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per model")
    parser.add_argument("--dataset_nr", type=int, default=0, help="Dataset number")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    
    args = parser.parse_args()
    main(args)
