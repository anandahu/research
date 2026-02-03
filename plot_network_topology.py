"""
Visualize and compare normal vs jammed network topology.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_dt_gcn.brite_loader import BRITELoader

# Load the topology
brite_path = r"c:\Users\ASUS\Downloads\research\RP15\50nodes.brite"

loader = BRITELoader(brite_path)
G = loader.load_graph()
pos = loader.get_node_positions()

# Simulate jammer positions (example: 2 jammers at specific locations)
np.random.seed(42)
jammer_positions = [(35, 60), (70, 25)]  # Two jammer locations
jam_radius = 20  # Nodes within this radius are jammed

# Calculate which nodes are jammed
def get_jammed_nodes(pos, jammer_positions, radius):
    jammed = set()
    affected = set()  # Partially affected
    for node, (x, y) in pos.items():
        for jx, jy in jammer_positions:
            dist = np.sqrt((x - jx)**2 + (y - jy)**2)
            if dist < radius:
                jammed.add(node)
            elif dist < radius * 1.5:
                affected.add(node)
    return jammed, affected - jammed

jammed_nodes, affected_nodes = get_jammed_nodes(pos, jammer_positions, jam_radius)

# Create side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left: Original (Unjammed) Topology
ax1 = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.4, edge_color='#3498db', width=1.5)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=400, node_color='#2ecc71', 
                       edgecolors='black', linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=7, font_weight='bold')
ax1.set_title('Normal Topology (No Jamming)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')

# Right: Jammed Topology
ax2 = axes[1]
nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.4, edge_color='#3498db', width=1.5)

# Color nodes based on jamming status
node_colors = []
for node in G.nodes():
    if node in jammed_nodes:
        node_colors.append('#e74c3c')  # Red for jammed
    elif node in affected_nodes:
        node_colors.append('#f39c12')  # Orange for partially affected
    else:
        node_colors.append('#2ecc71')  # Green for normal

nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=400, node_color=node_colors, 
                       edgecolors='black', linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax2, font_size=7, font_weight='bold')

# Draw jammer positions with big X markers
for jx, jy in jammer_positions:
    ax2.scatter(jx, jy, s=800, c='red', marker='X', zorder=5, 
                edgecolors='black', linewidths=2)
    # Draw jamming radius
    circle = plt.Circle((jx, jy), jam_radius, color='red', fill=False, 
                         linestyle='--', linewidth=2, alpha=0.5)
    ax2.add_patch(circle)

ax2.set_title(f'Jammed Topology ({len(jammed_nodes)} jammed, {len(affected_nodes)} affected)', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Normal Node'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Partially Affected'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Jammed Node'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
           markersize=15, markeredgecolor='black', label='Jammer Location')
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.suptitle('Network Topology: Normal vs Jammed', fontsize=18, fontweight='bold')
plt.tight_layout()

# Save
output_dir = r"c:\Users\ASUS\Downloads\research\integrated_dt_gcn\results"
output_path = os.path.join(output_dir, "topology_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Topology comparison saved to: {output_path}")
print(f"Jammed nodes: {sorted(jammed_nodes)}")
print(f"Affected nodes: {sorted(affected_nodes)}")

plt.show()
