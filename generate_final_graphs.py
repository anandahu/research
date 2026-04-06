"""
Generate final comparison graphs from latest run (20260305_104451).
Saves all graphs + CSV copies to 'final results' folder.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "integrated_dt_gcn", "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "final results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Latest run files
EPISODES_CSV = os.path.join(RESULTS_DIR, "comparison_episodes_20260305_104451.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "comparison_summary_20260305_104451.csv")
IMPROVEMENTS_CSV = os.path.join(RESULTS_DIR, "comparison_improvements.csv")

# Load data
print("Loading data...")
episodes_df = pd.read_csv(EPISODES_CSV)
summary_df = pd.read_csv(SUMMARY_CSV)

print(f"Episodes: {len(episodes_df)}")
print(f"\nSummary:")
print(summary_df.to_string())

# Extract columns
ma_rewards = episodes_df['Multi-Agent_GAT_DT_50_agents_reward'].values
bl_rewards = episodes_df['GAT_Baseline_4_features_reward'].values
ma_path = episodes_df['Multi-Agent_GAT_DT_50_agents_path_length'].values
bl_path = episodes_df['GAT_Baseline_4_features_path_length'].values
ma_jammed = episodes_df['Multi-Agent_GAT_DT_50_agents_jammed_steps'].values
bl_jammed = episodes_df['GAT_Baseline_4_features_jammed_steps'].values

# Summary values
ma_summary = summary_df[summary_df['model'].str.contains('Multi-Agent')].iloc[0]
bl_summary = summary_df[summary_df['model'].str.contains('Baseline')].iloc[0]

colors = ['#2ecc71', '#e74c3c']
labels = ['MA GAT+DT', 'GAT Baseline']
WINDOW = 50

# ============================================================
# 1. REWARD CONVERGENCE GRAPH
# ============================================================
print("\nGenerating reward convergence graph...")
fig, ax = plt.subplots(figsize=(12, 6))

ma_smooth = pd.Series(ma_rewards).rolling(WINDOW).mean()
bl_smooth = pd.Series(bl_rewards).rolling(WINDOW).mean()

ax.plot(ma_smooth, label=labels[0], color=colors[0], linewidth=2)
ax.plot(bl_smooth, label=labels[1], color=colors[1], linewidth=2)

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel(f'Reward ({WINDOW}-episode moving avg)', fontsize=13)
ax.set_title('Multi-Agent GAT+DT vs GAT Baseline: Reward Convergence', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_reward_graph.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_reward_graph.png")
plt.close()

# ============================================================
# 2. PATH LENGTH CONVERGENCE GRAPH
# ============================================================
print("Generating path length graph...")
fig, ax = plt.subplots(figsize=(12, 6))

ma_path_smooth = pd.Series(ma_path).rolling(WINDOW).mean()
bl_path_smooth = pd.Series(bl_path).rolling(WINDOW).mean()

ax.plot(ma_path_smooth, label=labels[0], color=colors[0], linewidth=2)
ax.plot(bl_path_smooth, label=labels[1], color=colors[1], linewidth=2)

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel(f'Path Length ({WINDOW}-episode moving avg)', fontsize=13)
ax.set_title('Path Length Comparison over 3,000 Episodes', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_path_length_graph.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_path_length_graph.png")
plt.close()

# ============================================================
# 3. JAMMED STEPS CONVERGENCE GRAPH
# ============================================================
print("Generating jammed steps graph...")
fig, ax = plt.subplots(figsize=(12, 6))

ma_j_smooth = pd.Series(ma_jammed).rolling(WINDOW).mean()
bl_j_smooth = pd.Series(bl_jammed).rolling(WINDOW).mean()

ax.plot(ma_j_smooth, label=labels[0], color=colors[0], linewidth=2)
ax.plot(bl_j_smooth, label=labels[1], color=colors[1], linewidth=2)

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel(f'Jammed Steps ({WINDOW}-episode moving avg)', fontsize=13)
ax.set_title('Jammed Steps Comparison over 3,000 Episodes', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_jammed_steps_graph.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_jammed_steps_graph.png")
plt.close()

# ============================================================
# 4. TRAINING TIME BAR CHART
# ============================================================
print("Generating training time chart...")
fig, ax = plt.subplots(figsize=(8, 6))

times = [ma_summary['training_time_min'], bl_summary['training_time_min']]
bars = ax.bar(labels, times, color=colors, edgecolor='black', linewidth=1.2, width=0.5)

for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{t:.2f} min', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Training Time (minutes)', fontsize=13)
ax.set_title('Training Time Comparison (3,000 Episodes)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(times) * 1.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_time_graph.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_time_graph.png")
plt.close()

# ============================================================
# 5. SUCCESS RATE & PDR BAR CHART
# ============================================================
print("Generating success rate chart...")
fig, ax = plt.subplots(figsize=(8, 6))

success_rates = [ma_summary['success_rate'] * 100, bl_summary['success_rate'] * 100]
bars = ax.bar(labels, success_rates, color=colors, edgecolor='black', linewidth=1.2, width=0.5)

for bar, val in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Success Rate (%)', fontsize=13)
ax.set_title('Success Rate Comparison (3,000 Episodes)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_success_rate_graph.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_success_rate_graph.png")
plt.close()

# ============================================================
# 6. COMBINED 2x2 SUMMARY FIGURE
# ============================================================
print("Generating combined summary figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Reward convergence
axes[0, 0].plot(ma_smooth, label=labels[0], color=colors[0], linewidth=2)
axes[0, 0].plot(bl_smooth, label=labels[1], color=colors[1], linewidth=2)
axes[0, 0].set_title('Reward Convergence', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Top-right: Performance bar chart
metrics_names = ['Success Rate', 'Avg Reward / 2', 'Jammed Steps / 5']
x = np.arange(len(metrics_names))
width = 0.35
values_ma = [ma_summary['success_rate'], ma_summary['avg_reward'] / 2, 
             float(ma_path[-100:].mean()) / 100]  # normalized
values_bl = [bl_summary['success_rate'], bl_summary['avg_reward'] / 2,
             float(bl_path[-100:].mean()) / 100]
axes[0, 1].bar(x - width/2, values_ma, width, label=labels[0], color=colors[0])
axes[0, 1].bar(x + width/2, values_bl, width, label=labels[1], color=colors[1])
axes[0, 1].set_title('Performance Comparison', fontweight='bold', fontsize=12)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metrics_names, fontsize=10)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Bottom-left: Path length
axes[1, 0].plot(ma_path_smooth, label=labels[0], color=colors[0], linewidth=2)
axes[1, 0].plot(bl_path_smooth, label=labels[1], color=colors[1], linewidth=2)
axes[1, 0].set_title('Path Length', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Hops')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: Summary text
axes[1, 1].axis('off')
summary_text = f"""
COMPARISON SUMMARY
==================

Multi-Agent GAT+DT (50 agents):
  • Success Rate: {ma_summary['success_rate']:.1%}
  • Avg Reward: {ma_summary['avg_reward']:.3f}
  • Avg Path Length: {ma_summary['avg_path_length']:.2f}
  • Avg Jammed Steps: {ma_summary['avg_jammed_steps']:.2f}
  • Training Time: {ma_summary['training_time_min']:.1f} min

GAT Baseline (single-agent):
  • Success Rate: {bl_summary['success_rate']:.1%}
  • Avg Reward: {bl_summary['avg_reward']:.3f}
  • Avg Path Length: {bl_summary['avg_path_length']:.2f}
  • Avg Jammed Steps: {bl_summary['avg_jammed_steps']:.2f}
  • Training Time: {bl_summary['training_time_min']:.1f} min

Improvement (MA over Baseline):
  • Success: {(ma_summary['success_rate'] - bl_summary['success_rate'])*100:+.1f} pp
  • Jammed: {(bl_summary['avg_jammed_steps'] - ma_summary['avg_jammed_steps']):+.2f} steps avoided
"""
axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Multi-Agent GAT+DT vs GAT Baseline', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_results.png"), dpi=200, bbox_inches='tight')
print("  Saved: comparison_results.png")
plt.close()

# ============================================================
# COPY CSVs
# ============================================================
print("\nCopying CSV files...")
csv_files = [
    (EPISODES_CSV, "comparison_episodes_20260305_104451.csv"),
    (SUMMARY_CSV, "comparison_summary_20260305_104451.csv"),
]

if os.path.exists(IMPROVEMENTS_CSV):
    csv_files.append((IMPROVEMENTS_CSV, "comparison_improvements.csv"))

for src, name in csv_files:
    dst = os.path.join(OUTPUT_DIR, name)
    shutil.copy2(src, dst)
    print(f"  Copied: {name}")

# Also copy the network topology image if it exists
network_img = os.path.join(BASE_DIR, "images", "network.jpeg")
if os.path.exists(network_img):
    shutil.copy2(network_img, os.path.join(OUTPUT_DIR, "network.jpeg"))
    print("  Copied: network.jpeg")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 60)
print("ALL FILES IN 'final results':")
print("=" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:<50s} {size/1024:>8.1f} KB")
print("=" * 60)
print("Done!")
