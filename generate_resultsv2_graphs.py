"""
Generate comparison graphs with new color scheme.
Saves all graphs to 'resultsv2' folder.
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
OUTPUT_DIR = os.path.join(BASE_DIR, "resultsv2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Latest run files
EPISODES_CSV = os.path.join(RESULTS_DIR, "comparison_episodes_20260305_104451.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "comparison_summary_20260305_104451.csv")

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

# NEW COLOR SCHEME — Royal Blue & Burnt Orange
colors = ['#1a73e8', '#e8710a']
labels = ['MA GAT+DT', 'GAT Baseline']
WINDOW = 50

# ============================================================
# 1. REWARD CONVERGENCE GRAPH
# ============================================================
print("\nGenerating reward convergence graph...")
fig, ax = plt.subplots(figsize=(12, 6))

ma_smooth = pd.Series(ma_rewards).rolling(WINDOW).mean()
bl_smooth = pd.Series(bl_rewards).rolling(WINDOW).mean()

ax.plot(ma_smooth, label=labels[0], color=colors[0], linewidth=2.5)
ax.plot(bl_smooth, label=labels[1], color=colors[1], linewidth=2.5)

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

ax.plot(ma_path_smooth, label=labels[0], color=colors[0], linewidth=2.5)
ax.plot(bl_path_smooth, label=labels[1], color=colors[1], linewidth=2.5)

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

ax.plot(ma_j_smooth, label=labels[0], color=colors[0], linewidth=2.5)
ax.plot(bl_j_smooth, label=labels[1], color=colors[1], linewidth=2.5)

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
# 5. SUCCESS RATE BAR CHART
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
# COPY network topology image
# ============================================================
network_img = os.path.join(BASE_DIR, "images", "network.jpeg")
if not os.path.exists(network_img):
    network_img = os.path.join(BASE_DIR, "network.jpeg")
if os.path.exists(network_img):
    shutil.copy2(network_img, os.path.join(OUTPUT_DIR, "network.jpeg"))
    print("  Copied: network.jpeg")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 60)
print("ALL FILES IN 'resultsv2':")
print("=" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:<50s} {size/1024:>8.1f} KB")
print("=" * 60)
print("Done!")
