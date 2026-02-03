# BRITE Topology File Documentation

## Overview

The `50nodes.brite` file defines the **network topology** for the mesh routing system. BRITE (Boston university Representative Internet Topology gEnerator) is a standard format for representing network graphs with nodes and edges.

---

## 1. File Summary

| Property | Value |
|----------|-------|
| **File Location** | `RP15/50nodes.brite` |
| **File Size** | 9,147 bytes |
| **Total Lines** | 158 |
| **Nodes** | 50 |
| **Edges** | 100 |
| **Model** | ASWaxman (Waxman random graph for AS-level) |

---

## 2. File Structure

```
Line 1:      Topology header
Line 2:      Model parameters
Line 3:      Empty
Line 4:      Nodes section header
Lines 5-54:  Node definitions (50 nodes)
Lines 55-56: Empty
Line 57:     Edges section header
Lines 58-157: Edge definitions (100 edges)
Line 158:    Empty
```

---

## 3. Header Section

### Line 1: Topology Summary

```
Topology: ( 50 Nodes, 100 Edges )
```

### Line 2: Model Parameters

```
Model (3 - ASWaxman):  50 100 10 1  2  0.15000000596046448 0.20000000298023224 1 2 1.0 99.0
```

| Position | Value | Description |
|----------|-------|-------------|
| Model ID | 3 | ASWaxman model |
| N | 50 | Number of nodes |
| E | 100 | Number of edges |
| HS | 10 | Size of plane (10×10 squares) |
| LS | 1 | Link selection type |
| NodePlacement | 2 | Random placement |
| α | 0.15 | Waxman alpha parameter |
| β | 0.20 | Waxman beta parameter |
| m | 1 | Lower bound of edge degree |
| M | 2 | Upper bound of edge degree |
| MinCoord | 1.0 | Minimum coordinate |
| MaxCoord | 99.0 | Maximum coordinate |

### Waxman Model

The Waxman model creates edges between nodes with probability:
```
P(u,v) = β × exp(-d(u,v) / (L × α))

Where:
  d(u,v) = Euclidean distance between nodes u and v
  L = maximum distance between any two nodes
  α = 0.15 (controls distance importance)
  β = 0.20 (controls overall edge density)
```

---

## 4. Node Section

### Header
```
Nodes: ( 50 )
```

### Node Format
```
node_id    x_coord    y_coord    in_degree    out_degree    AS_id    type
```

### Field Descriptions

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `node_id` | `int` | 0-49 | Unique node identifier |
| `x_coord` | `int` | 1-99 | X position in grid |
| `y_coord` | `int` | 1-99 | Y position in grid |
| `in_degree` | `int` | 2-10 | Number of incoming edges |
| `out_degree` | `int` | 2-10 | Number of outgoing edges |
| `AS_id` | `int` | 0-49 | Autonomous System ID (same as node_id) |
| `type` | `str` | `AS_NODE` | Node type |

### Sample Nodes

```
0    51    25    7    7    0    AS_NODE
1    19    18    6    6    1    AS_NODE
2     2    12    6    6    2    AS_NODE
3    10    34   10   10    3    AS_NODE
...
49   37    84    2    2   49    AS_NODE
```

### Node Position Distribution

| Statistic | X Coordinate | Y Coordinate |
|-----------|--------------|--------------|
| Minimum | 1 | 2 |
| Maximum | 99 | 96 |
| Mean | ~47 | ~46 |
| Std Dev | ~29 | ~29 |

### Complete Node Table

| Node | X | Y | Degree | Node | X | Y | Degree |
|------|---|---|--------|------|---|---|--------|
| 0 | 51 | 25 | 7 | 25 | 40 | 96 | 2 |
| 1 | 19 | 18 | 6 | 26 | 42 | 50 | 3 |
| 2 | 2 | 12 | 6 | 27 | 54 | 6 | 2 |
| 3 | 10 | 34 | 10 | 28 | 65 | 93 | 2 |
| 4 | 33 | 9 | 9 | 29 | 20 | 79 | 2 |
| 5 | 92 | 56 | 10 | 30 | 90 | 45 | 4 |
| 6 | 74 | 28 | 6 | 31 | 52 | 49 | 4 |
| 7 | 21 | 52 | 7 | 32 | 30 | 69 | 2 |
| 8 | 11 | 19 | 8 | 33 | 33 | 3 | 2 |
| 9 | 53 | 12 | 8 | 34 | 6 | 36 | 2 |
| 10 | 94 | 59 | 6 | 35 | 42 | 71 | 2 |
| 11 | 17 | 67 | 5 | 36 | 99 | 20 | 3 |
| 12 | 89 | 82 | 6 | 37 | 82 | 11 | 2 |
| 13 | 97 | 16 | 3 | 38 | 31 | 91 | 2 |
| 14 | 90 | 92 | 3 | 39 | 10 | 15 | 3 |
| 15 | 94 | 3 | 3 | 40 | 55 | 89 | 3 |
| 16 | 22 | 82 | 6 | 41 | 20 | 82 | 2 |
| 17 | 26 | 2 | 5 | 42 | 57 | 20 | 3 |
| 18 | 25 | 60 | 4 | 43 | 87 | 92 | 2 |
| 19 | 70 | 85 | 5 | 44 | 63 | 80 | 2 |
| 20 | 78 | 41 | 5 | 45 | 86 | 35 | 2 |
| 21 | 74 | 65 | 5 | 46 | 25 | 8 | 3 |
| 22 | 59 | 89 | 2 | 47 | 18 | 36 | 2 |
| 23 | 19 | 27 | 2 | 48 | 1 | 7 | 2 |
| 24 | 42 | 40 | 3 | 49 | 37 | 84 | 2 |

---

## 5. Edge Section

### Header
```
Edges: ( 100 )
```

### Edge Format
```
edge_id    src    dst    delay    bandwidth    length    AS_from    AS_to    type    direction
```

### Field Descriptions

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `edge_id` | `int` | 0-99 | Unique edge identifier |
| `src` | `int` | 0-49 | Source node ID |
| `dst` | `int` | 0-49 | Destination node ID |
| `delay` | `float` | 3.6-92.4 | Edge delay/latency (ms) |
| `bandwidth` | `float` | 0.01-0.31 | Edge bandwidth (normalized) |
| `length` | `float` | 1.9-99.6 | Physical link length |
| `AS_from` | `int` | 0-49 | Source AS (same as src) |
| `AS_to` | `int` | 0-49 | Destination AS (same as dst) |
| `type` | `str` | `E_AS` | Edge type |
| `direction` | `str` | `U` | Undirected |

### Sample Edges

```
0    2    1    18.027756377319946    0.060134122444534435    28.737834920488602    2    1    E_AS    U
1    2    0    50.695167422546305    0.16910087652220493     25.609067055706245    2    0    E_AS    U
...
99   1   36    80.02499609497022     0.26693465415654394     92.30474409244215     1   36    E_AS    U
```

### Edge Statistics

| Metric | Delay (ms) | Bandwidth | Length |
|--------|------------|-----------|--------|
| Minimum | 3.61 | 0.012 | 1.94 |
| Maximum | 92.44 | 0.308 | 99.62 |
| Mean | 35.6 | 0.119 | 47.3 |
| Std Dev | 20.1 | 0.067 | 30.4 |

### Degree Distribution

| Degree | Count | Nodes |
|--------|-------|-------|
| 2 | 19 | 22, 23, 27, 28, 29, 32, 33, 34, 35, 37, 38, 41, 43, 44, 45, 47, 48, 49 |
| 3 | 9 | 13, 15, 24, 26, 36, 39, 40, 42, 46 |
| 4 | 3 | 18, 30, 31 |
| 5 | 5 | 11, 17, 19, 20, 21 |
| 6 | 6 | 1, 2, 6, 10, 12, 16 |
| 7 | 3 | 0, 7, 8 |
| 8 | 2 | 8, 9 |
| 9 | 1 | 4 |
| 10 | 2 | 3, 5 |

---

## 6. Usage in Code

### Loading with BRITELoader

```python
from integrated_dt_gcn.brite_loader import BRITELoader

loader = BRITELoader("RP15/50nodes.brite")
graph = loader.load_graph()

print(f"Nodes: {graph.number_of_nodes()}")  # 50
print(f"Edges: {graph.number_of_edges()}")  # 100

# Access node position
pos = graph.nodes[0]['pos']
print(f"Node 0 position: {pos}")  # (51, 25)

# Access edge weight
weight = graph[2][1]['weight']
capacity = graph[2][1]['capacity']
print(f"Edge 2→1: weight={weight:.2f}, capacity={capacity:.4f}")
```

### Loading with util.create_graph()

```python
from RP15.environment.util import create_graph

G = create_graph(50, 100, "50nodes.brite")

# Graph has:
# - 50 nodes with 'pos' attribute
# - 100 edges with 'weight' and 'capacity' attributes
```

### Direct Parsing Example

```python
import networkx as nx

def load_brite(filepath):
    """Manually parse BRITE file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0]  # "Topology: ( 50 Nodes, 100 Edges )"
    
    g = nx.Graph()
    
    # Parse nodes (lines 5-54, 0-indexed: 4-53)
    for i in range(4, 54):
        parts = lines[i].strip().split('\t')
        node_id = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        g.add_node(node_id, pos=(x, y))
    
    # Parse edges (lines 58-157, 0-indexed: 57-156)
    for i in range(57, 157):
        parts = lines[i].strip().split('\t')
        src = int(parts[1])
        dst = int(parts[2])
        delay = float(parts[3])
        bandwidth = float(parts[4])
        g.add_edge(src, dst, weight=delay, capacity=bandwidth)
    
    return g
```

---

## 7. Coordinate Scaling

### BRITE to Digital Twin Coordinates

The BRITE file uses coordinates 0-99, while the Digital Twin radio maps use 0-40.

```python
# Scaling factor
COORD_SCALE = 0.4  # 99 * 0.4 ≈ 40

def scale_to_dt(brite_coord):
    """Convert BRITE coord to DT coord."""
    return min(40, int(brite_coord * 0.4))

# Examples:
# Node 0: (51, 25) → (20, 10)
# Node 5: (92, 56) → (36, 22)
# Node 48: (1, 7) → (0, 2)
```

### Scaled Position Map

```python
from integrated_dt_gcn.brite_loader import BRITELoader

loader = BRITELoader("RP15/50nodes.brite")
loader.load_graph()
scaled = loader.get_scaled_positions()

# scaled[0] = (20, 10)  # Node 0
# scaled[5] = (36, 22)  # Node 5
# etc.
```

---

## 8. Graph Visualization

### NetworkX Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

# Load graph
G = load_brite("50nodes.brite")

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Draw
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, 
        node_color='lightblue', 
        node_size=500,
        font_size=8,
        edge_color='gray')
plt.title("50-Node Mesh Network Topology")
plt.savefig("network_topology.png")
plt.show()
```

### Edge Weight Visualization

```python
# Color edges by weight
weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw(G, pos, 
        edge_color=weights, 
        edge_cmap=plt.cm.Reds,
        width=2)
plt.colorbar(label='Edge Delay (ms)')
```

---

## 9. Network Properties

### Graph Metrics

| Metric | Value |
|--------|-------|
| Nodes | 50 |
| Edges | 100 |
| Density | 0.0816 |
| Average Degree | 4.0 |
| Diameter | ~8 hops |
| Average Path Length | ~3.5 hops |
| Clustering Coefficient | ~0.15 |
| Is Connected | Yes |

### Centrality Analysis

```python
import networkx as nx

# Degree centrality
deg_cent = nx.degree_centrality(G)
most_connected = max(deg_cent, key=deg_cent.get)
# Node 3 and 5 have highest degree (10)

# Betweenness centrality  
bet_cent = nx.betweenness_centrality(G)
most_important = max(bet_cent, key=bet_cent.get)
# Identifies nodes on many shortest paths

# Closeness centrality
close_cent = nx.closeness_centrality(G)
most_central = max(close_cent, key=close_cent.get)
# Identifies nodes closest to all others
```

---

## 10. Edge Weight Interpretation

### Weight (Delay)

- **Units**: Milliseconds (loosely interpreted)
- **Range**: 3.6 to 92.4
- **Usage**: Lower is better for routing
- **In Code**: `graph[u][v]['weight']`

### Capacity (Bandwidth)

- **Units**: Normalized (0-1)
- **Range**: 0.012 to 0.308
- **Usage**: Higher is better for throughput
- **In Code**: `graph[u][v]['capacity']`

### Normalization in Training

```python
# RP15 normalization
capacity = float(line[5]) / 100  # Divide by 100

# Integrated module: DT updates these dynamically
# based on radio map RSS values
```

---

## 11. Relationship to DT Integration

### Link Weight Updates

```python
# In HybridEnv.reset():
radio_map = loader.get_radio_map_array(scenario_idx)
mesh_twin.update_link_weights(radio_map)

# This REPLACES the static BRITE weights with
# DT-derived weights based on signal strength
```

### Feature Computation

```python
# Features 2 and 3 use edge weights:
x[node, 2] = mean(graph[node][n]['weight'] for n in neighbors)
x[node, 3] = mean(graph[node][n]['capacity'] for n in neighbors)

# With DT: these reflect current radio conditions
# Without DT: these are static BRITE values
```

---

## 12. Alternative Topologies

The repository uses only `50nodes.brite`, but BRITE can generate other topologies:

### Common BRITE Models

| Model | Description |
|-------|-------------|
| Waxman | Distance-based random edges |
| BA | Barabási-Albert (preferential attachment) |
| GLP | Generalized Linear Preference |
| AS | Autonomous System level |
| Router | Router level |

### Generating New Topology

```bash
# Using BRITE tool (external)
java -jar BRITE.jar -f config.txt -o new_topology.brite
```

### Adapting Code for Different Sizes

```python
# In config.py
NUM_NODES = 100  # Change from 50
NUM_EDGES = 200

# In brite_loader.py
# Parsing logic is automatic based on header
```
