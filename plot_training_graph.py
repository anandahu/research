"""
Generate training result graphs from episode_metrics CSV.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to training results
results_dir = r"c:\Users\ASUS\Downloads\research\integrated_dt_gcn\results"
training_csv = os.path.join(results_dir, "episode_metrics_20260203_034847.csv")


def plot_training_reward():
    """Plot training reward over episodes."""
    df = pd.read_csv(training_csv)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['episode'], df['episode_reward'], alpha=0.3, color='#3498db', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['episode_reward'].rolling(100).mean(), 
            color='#2ecc71', linewidth=2, label='100-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Training Reward Convergence (GCN + DT)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "training_reward_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training reward graph saved to: {output_path}")
    plt.close()


def plot_training_path_length():
    """Plot path length over episodes."""
    df = pd.read_csv(training_csv)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['episode'], df['path_length'], alpha=0.3, color='#9b59b6', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['path_length'].rolling(100).mean(), 
            color='#e74c3c', linewidth=2, label='100-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Path Length (hops)', fontsize=12)
    ax.set_title('Path Length Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "training_path_length_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training path length graph saved to: {output_path}")
    plt.close()


def plot_training_jammed_steps():
    """Plot jammed steps over episodes."""
    df = pd.read_csv(training_csv)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['episode'], df['jammed_steps'], alpha=0.3, color='#e67e22', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['jammed_steps'].rolling(100).mean(), 
            color='#c0392b', linewidth=2, label='100-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Jammed Steps', fontsize=12)
    ax.set_title('Jammed Steps Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "training_jammed_steps_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training jammed steps graph saved to: {output_path}")
    plt.close()


def plot_training_latency():
    """Plot total latency over episodes."""
    df = pd.read_csv(training_csv)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['episode'], df['total_latency'], alpha=0.3, color='#1abc9c', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['total_latency'].rolling(100).mean(), 
            color='#16a085', linewidth=2, label='100-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Latency (ms)', fontsize=12)
    ax.set_title('Total Latency Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "training_latency_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training latency graph saved to: {output_path}")
    plt.close()


def plot_training_combined():
    """Plot combined training metrics in subplots."""
    df = pd.read_csv(training_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward
    ax = axes[0, 0]
    ax.plot(df['episode'], df['episode_reward'].rolling(100).mean(), color='#2ecc71', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Path Length
    ax = axes[0, 1]
    ax.plot(df['episode'], df['path_length'].rolling(100).mean(), color='#e74c3c', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Path Length')
    ax.set_title('Path Length', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Jammed Steps
    ax = axes[1, 0]
    ax.plot(df['episode'], df['jammed_steps'].rolling(100).mean(), color='#c0392b', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Jammed Steps')
    ax.set_title('Jammed Steps', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Latency
    ax = axes[1, 1]
    ax.plot(df['episode'], df['total_latency'].rolling(100).mean(), color='#16a085', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Total Latency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics (100-episode moving avg)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, "training_combined_graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training combined graph saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_training_reward()
    plot_training_path_length()
    plot_training_jammed_steps()
    plot_training_latency()
    plot_training_combined()
    print("\nAll training graphs generated!")
