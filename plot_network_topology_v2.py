"""
Visualize normal network topology — v2 colors.
Saves to resultsv2 folder.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_dt_gcn.brite_loader import BRITELoader

brite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RP15", "50nodes.brite")

loader = BRITELoader(brite_path)
G = loader.load_graph()
pos = loader.get_node_positions()

CLR_NODE = '#1a73e8'
CLR_EDGE = '#7b8794'

fig, ax = plt.subplots(figsize=(10, 10))

nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color=CLR_EDGE, width=1.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400, node_color=CLR_NODE,
                       edgecolors='black', linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold', font_color='white')

ax.set_title(f'Simulated WMN Topology: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultsv2")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "network.jpeg")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to: {output_path}")
plt.close()
