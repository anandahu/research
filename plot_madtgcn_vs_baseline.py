
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(
    r"c:\Users\ASUS\Downloads\research\integrated_dt_gcn\comparison and training results\comparison_episodes_20260218_104707.csv"
)

ma_col   = "Multi-Agent_GCN_DT_50_agents"
base_col = "GCN_Baseline_4_features"

ma_reward   = df[f"{ma_col}_reward"]
base_reward = df[f"{base_col}_reward"]
ma_path     = df[f"{ma_col}_path_length"]
base_path   = df[f"{base_col}_path_length"]
ma_jammed   = df[f"{ma_col}_jammed_steps"]
base_jammed = df[f"{base_col}_jammed_steps"]
ma_latency  = df[f"{ma_col}_latency"]
base_latency= df[f"{base_col}_latency"]
episodes    = df["episode"]

def smooth(y, window=100):
    return pd.Series(y).rolling(window, min_periods=1).mean()

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

MA_COLOR   = "#2196F3"   # blue
BASE_COLOR = "#FF5722"   # orange-red
ALPHA_RAW  = 0.12
ALPHA_SM   = 1.0

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle(
    "Multi-Agent DT-GCN  vs  GCN Baseline — 3000 Episode Training Comparison",
    fontsize=16, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

metrics = [
    (gs[0, 0], ma_reward,   base_reward,   "Episode Reward",       "Reward"),
    (gs[0, 1], ma_path,     base_path,     "Path Length (Hops)",   "Hops"),
    (gs[1, 0], ma_jammed,   base_jammed,   "Jammed Steps",         "Steps"),
    (gs[1, 1], ma_latency,  base_latency,  "End-to-End Latency",   "Latency"),
]

for spec, ma_y, base_y, title, ylabel in metrics:
    ax = fig.add_subplot(spec)
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Raw (faint)
    ax.plot(episodes, ma_y,   color=MA_COLOR,   alpha=ALPHA_RAW, linewidth=0.6)
    ax.plot(episodes, base_y, color=BASE_COLOR, alpha=ALPHA_RAW, linewidth=0.6)

    # Smoothed
    ax.plot(episodes, smooth(ma_y),   color=MA_COLOR,   linewidth=2.2,
            label="MA DT-GCN (50 agents)")
    ax.plot(episodes, smooth(base_y), color=BASE_COLOR, linewidth=2.2,
            label="GCN Baseline")

    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Episode", color="#aaaacc", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaacc", fontsize=10)
    ax.tick_params(colors="#aaaacc")
    ax.grid(color="#333355", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

# ── Summary bar chart ──────────────────────────────────────────────────────
summary_data = {
    "Avg Reward":       (-0.741, -7.334),
    "Success Rate (%)": (54,      85),
    "Avg Path (hops)":  (7.2,     23.63),
    "Avg Jammed":       (0.92,    0.94),
}

# Inset axes for summary
ax_bar = fig.add_axes([0.15, -0.02, 0.70, 0.18])
ax_bar.set_facecolor("#1a1a2e")
for spine in ax_bar.spines.values():
    spine.set_edgecolor("#333355")

labels = list(summary_data.keys())
ma_vals   = [v[0] for v in summary_data.values()]
base_vals = [v[1] for v in summary_data.values()]

x = np.arange(len(labels))
w = 0.32
bars1 = ax_bar.bar(x - w/2, ma_vals,   w, label="MA DT-GCN",    color=MA_COLOR,   alpha=0.85)
bars2 = ax_bar.bar(x + w/2, base_vals, w, label="GCN Baseline", color=BASE_COLOR, alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h >= 0 else -2),
                f"{h}", ha="center", va="bottom", fontsize=8, color="white")
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h >= 0 else -2),
                f"{h}", ha="center", va="bottom", fontsize=8, color="white")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, color="#aaaacc", fontsize=9)
ax_bar.set_title("Aggregate Performance Summary", color="white",
                  fontsize=11, fontweight="bold", pad=6)
ax_bar.tick_params(colors="#aaaacc")
ax_bar.grid(axis="y", color="#333355", linestyle="--", alpha=0.4)
ax_bar.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")
ax_bar.axhline(0, color="#555577", linewidth=0.8)

out_path = r"c:\Users\ASUS\Downloads\research\integrated_dt_gcn\comparison and training results\madtgcn_vs_baseline_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
