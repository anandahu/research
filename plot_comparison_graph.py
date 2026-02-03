"""
Generate comparison graphs from CSV data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
results_dir = r"c:\Users\ASUS\Downloads\research\integrated_dt_gcn\results"
episodes_csv = os.path.join(results_dir, "comparison_episodes_20260203_013945.csv")
summary_csv = os.path.join(results_dir, "comparison_summary_20260203_013945.csv")


def plot_reward_comparison():
    """Plot reward line graph."""
    df = pd.read_csv(episodes_csv)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['episode'], df['GCN_+_DT_7_features_reward'].rolling(50).mean(), 
            label='GCN + DT (7 features)', color='#2ecc71', linewidth=2)
    ax.plot(df['episode'], df['GCN_Baseline_4_features_reward'].rolling(50).mean(), 
            label='GCN Baseline (4 features)', color='#e74c3c', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (50-episode moving avg)', fontsize=12)
    ax.set_title('GCN + Digital Twin vs Baseline: Training Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(df))
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "comparison_reward_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Reward graph saved to: {output_path}")
    plt.close()


def plot_time_comparison():
    """Plot training time bar chart."""
    df = pd.read_csv(summary_csv)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['GCN + DT', 'GCN Baseline']
    times = df['training_time_min'].tolist()
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(models, times, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time:.1f} min', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison (3000 Episodes)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "comparison_time_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Time graph saved to: {output_path}")
    plt.close()


def plot_pdr_success_comparison():
    """Plot PDR and Success Rate grouped bar chart."""
    df = pd.read_csv(summary_csv)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    import numpy as np
    
    models = ['GCN + DT', 'GCN Baseline']
    success_rates = (df['success_rate'] * 100).tolist()  # Convert to percentage
    pdr_values = (df['avg_pdr'] * 100).tolist()  # Convert to percentage
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, pdr_values, width, label='PDR (%)', 
                   color='#9b59b6', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars1, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, pdr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Success Rate & PDR Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "comparison_pdr_success_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"PDR/Success graph saved to: {output_path}")
    plt.close()


def plot_path_length_comparison():
    """Plot average path length bar chart."""
    df = pd.read_csv(summary_csv)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['GCN + DT', 'GCN Baseline']
    path_lengths = df['avg_path_length'].tolist()
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(models, path_lengths, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, length in zip(bars, path_lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{length:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Average Path Length (hops)', fontsize=12)
    ax.set_title('Average Path Length Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(path_lengths) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "comparison_path_length_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Path length graph saved to: {output_path}")
    plt.close()


def plot_jammed_steps_comparison():
    """Plot average jammed steps bar chart."""
    df = pd.read_csv(summary_csv)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['GCN + DT', 'GCN Baseline']
    jammed_steps = df['avg_jammed_steps'].tolist()
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(models, jammed_steps, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, steps in zip(bars, jammed_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{steps:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Average Jammed Steps per Episode', fontsize=12)
    ax.set_title('Average Jammed Steps Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(jammed_steps) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "comparison_jammed_steps_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Jammed steps graph saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_reward_comparison()
    plot_time_comparison()
    plot_pdr_success_comparison()
    plot_path_length_comparison()
    plot_jammed_steps_comparison()
    print("\nAll graphs generated!")

