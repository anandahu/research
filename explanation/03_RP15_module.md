# RP15 Module Deep Dive

## Module Overview

The `RP15/` module is the **baseline GNN-based routing system**. It implements Multi-Agent DQN with Graph Attention Networks (GAT) and GraphSAGE for network routing optimization. This module provides the foundational routing algorithms that are enhanced by the Digital Twin integration.

---

## Folder Structure

```
RP15/
├── 50nodes.brite           # Network topology definition
├── config.py               # Device configuration
├── train.py                # Main training script
├── environment/
│   ├── GCN_env.py          # GCN-compatible Gym environment
│   ├── env.py              # Base link-hop environment
│   └── util.py             # Utility functions (rewards, graph ops)
├── models/
│   ├── DQN.py              # Base DQN agent and MultiAgent
│   ├── GCN.py              # GAT-based model and agent
│   └── GraphSAGE.py        # GraphSAGE-based alternative
└── helper/
    └── graph.py            # Graph utility functions
```

---

## File-by-File Analysis

---

## 1. `50nodes.brite`

### Purpose
Network topology definition file in BRITE format (Boston university Representative Internet Topology gEnerator).

### Format
```
Topology: ( 50 Nodes, 100 Edges )

Model (3 - Loss):   ...

Nodes: (50)
0	31	42	3	3	-1	RT_NODE
1	88	94	3	3	-1	RT_NODE
...

Edges: (100)
0	0	1	64.03	10.01	1.72	-1	-1	E_RT	U
1	0	6	90.23	10.01	1.72	-1	-1	E_RT	U
...
```

### Node Format
```
node_id \t x_coord \t y_coord \t in_degree \t out_degree \t AS_id \t type
```

### Edge Format
```
edge_id \t src \t dst \t delay \t bandwidth \t length \t AS_from \t AS_to \t type \t direction
```

### Parsed Values
- **Nodes**: 50 nodes with positions (0-99, 0-99)
- **Edges**: 100 edges with:
  - `weight`: delay/latency value (normalized to 0-1 during loading)
  - `capacity`: bandwidth value (normalized to 0-1)

---

## 2. `config.py`

### Purpose
Simple device configuration.

### Content
```python
device = "cpu"
```

### Note
Unlike `integrated_dt_gcn/config.py`, this uses CPU by default. The integrated version auto-detects CUDA.

---

## 3. `train.py`

### Purpose
Main training script that demonstrates usage of different routing algorithms.

### Dependencies
- `environment.GCN_env.Env`: GCN environment
- `models.GraphSAGE.GraphSAGEPolicy, GCN_Agent`: GraphSAGE model
- `models.DQN.MultiAgent`: Multi-agent DQN baseline
- `environment.util`: Graph utilities

### Global Variables
| Variable | Type | Value | Description |
|----------|------|-------|-------------|
| `lwr` | `float` | `-1.51` | Negative reward threshold |
| `NODE_FEATURE_DIM` | `int` | `4` | Node features (no DT) |

### Function: `ma()`
**Purpose**: Train Multi-Agent DQN (one agent per node).

### Function: `gcn()`
**Purpose**: Train GAT-based centralized model.

### Function: `graph_sage()`
**Purpose**: Train GraphSAGE-based centralized model.

**Implementation**:
```python
def graph_sage():
    num_nodes = 50
    max_neighbors = 50

    environment = Env("training_data/save_file",
                      num_nodes_in_graph=num_nodes, 
                      max_neighbors=max_neighbors,
                      graph=G)

    policy_net = GraphSAGEPolicy(
        num_nodes=num_nodes,
        out_dim=max_neighbors,
        node_feat_dim=NODE_FEATURE_DIM  # 4
    ).to(device)

    target_net = GraphSAGEPolicy(
        num_nodes=num_nodes,
        out_dim=max_neighbors,
        node_feat_dim=NODE_FEATURE_DIM
    ).to(device)

    gcn = GCN_Agent(outputs=num_nodes, ...)
    gcn.run(300)
```

### Function: `spf()`
**Purpose**: Shortest Path First baseline evaluation.

### Function: `ecmp()`
**Purpose**: Equal Cost Multi-Path baseline evaluation.

---

## 4. `environment/GCN_env.py`

### Purpose
Gym environment for GCN-based routing with **4-dimensional node features**.

### Dependencies
- `gym`: RL environment
- `torch`, `torch_geometric`: Tensor operations
- `networkx`: Graph operations
- `config.device`: Computation device

### Constants
```python
NODE_FEATURE_DIM = 4  # vs 7 in integrated
```

### Class: `Env`

#### Constructor
```python
def __init__(self, save_file: str, num_nodes_in_graph: int = 5, 
             max_neighbors=5, graph=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_file` | `str` | Path to CSV log file |
| `num_nodes_in_graph` | `int` | Number of nodes |
| `max_neighbors` | `int` | Max neighbors per node |
| `graph` | `nx.Graph` | Pre-loaded graph (optional) |

#### Key Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.Graph` | Network topology |
| `edge_list` | `torch.Tensor` | Edge index for PyG |
| `source` | `int` | Current source node |
| `target` | `int` | Current target node |
| `current_node` | `int` | Agent's current position |
| `path` | `list` | Path taken so far |

#### Method: `compute_node_features()`

**Purpose**: Compute 4-dimensional features (baseline, no DT).

**Output**: `torch.Tensor (num_nodes, 4)`

**Features**:
```python
x[node, 0] = 1.0 if node == current_node else 0.0  # is_source
x[node, 1] = 1.0 if node == target else 0.0         # is_dest
x[node, 2] = mean(edge_weight for neighbors)        # avg_latency
x[node, 3] = mean(edge_capacity for neighbors)      # avg_bandwidth
```

**Comparison to Integrated**:
| Index | RP15 (4-dim) | Integrated (7-dim) |
|-------|--------------|-------------------|
| 0 | is_source | is_source |
| 1 | is_dest | is_dest |
| 2 | avg_latency | avg_latency |
| 3 | avg_bandwidth | avg_bandwidth |
| 4 | - | anomaly_score |
| 5 | - | jam_probability |
| 6 | - | neighbor_jam_avg |

#### Method: `step(action)`

**Output**: `Tuple[Data, float, bool, dict]`

**Algorithm**:
```
1. Validate action
2. Append node to path
3. Update current_node
4. Compute reward via _get_reward()
5. Record data if done
6. Return (observation, reward, done, info)
```

#### Method: `reset()`

**Algorithm**:
```
1. Get new random source-target pair
2. Reset path to [source]
3. Reset counters
4. Compute node features
5. Return (observation, info)
```

#### Method: `get_valid_actions()`

**Output**: `torch.Tensor (1, num_nodes)` - Boolean mask

**Logic**: True for neighbor nodes, False otherwise.

---

## 5. `environment/util.py`

### Purpose
Utility functions for graph operations and reward computation.

### Function: `create_graph(numNodes, numEdges, fileName)`

**Purpose**: Parse BRITE file and create NetworkX graph.

**Implementation**:
```python
def create_graph(numNodes=100, numEdges=200, fileName="Waxman.brite"):
    f = open(fileName)
    # Skip header lines
    for i in range(1, 5):
        f.readline()
    
    g = nx.Graph()
    
    # Parse nodes
    for i in range(numNodes):
        line = f.readline().strip().split("\t")
        g.add_node(i, pos=(int(line[1]), int(line[2])))
    
    # Skip edge header
    for i in range(3):
        f.readline()
    
    # Parse edges
    for i in range(numEdges):
        line = f.readline().strip().split("\t")
        g.add_edge(int(line[1]), int(line[2]),
                   weight=float(line[4]), 
                   capacity=float(line[5])/100)
    
    return g
```

### Function: `get_new_route(graph)`

**Purpose**: Generate random source-target pair with valid path.

**Output**: `Tuple[int, int]` - (source, target)

**Algorithm**:
```
while not done:
    node1 = random_choice(nodes)
    node2 = random_choice(nodes)
    if node1 != node2 and path_exists(node1, node2):
        done = True
return (node1, node2)
```

### Function: `compute_reward(graph, target, path)`

**Purpose**: Core reward function for routing.

**Output**: `Tuple[list, bool]` - ([rewards], done)

**Reward Logic**:
```python
def compute_reward(graph, target, path):
    c2 = astar_path_length(graph, path[-2], target)
    
    if path[-1] == target:
        # Reached destination
        actual_path_length = compute_path_length(graph, path)
        actual_flow_value = compute_flow_value(graph, path)
        
        if c2 == compute_path_length(graph, path[-2:]):
            # Optimal last hop
            return [1.01, actual_path_length, actual_flow_value], True
        else:
            # Suboptimal last hop
            return [-1.51, actual_path_length, actual_flow_value], True
    
    if len(path) > 10 * num_nodes:
        # Timeout
        return [-1, 0, 0], True
    
    # Progress reward
    c1 = astar_path_length(graph, path[-1], target)
    if c1 < c2:
        return [c2 - c1], False  # Positive progress
    return [-1], False          # Negative progress
```

### Function: `adjust_lat_band(graph, paths)`

**Purpose**: Simulate traffic by adjusting edge weights.

**Algorithm**:
```
For each path:
    For each edge in path:
        # Increase latency (congestion)
        graph[u][v]["weight"] = weight ** 0.3
        # Decrease bandwidth (usage)
        graph[u][v]["capacity"] = capacity ** 1.2
```

### Function: `get_max_neighbors(graph)`

**Purpose**: Find maximum degree in graph.

---

## 6. `environment/env.py`

### Purpose
Base link-hop environment (predecessor to GCN_env).

This file provides a simpler environment that returns raw observations rather than PyG Data objects. Used primarily by the MultiAgent approach.

---

## 7. `models/DQN.py`

### Purpose
Base DQN components: agent class, replay memory, and multi-agent framework.

### Class: `Transition`

```python
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
```

### Class: `ReplayMemory`

**Purpose**: Experience replay buffer with random memory.

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `capacity` | `int` | Maximum buffer size |
| `memory` | `list` | Main memory buffer |
| `random_memory` | `list` | Early random transitions |
| `position` | `int` | Current write position |

#### Methods
- `push(*args)`: Store transition
- `sample(batch_size)`: Random sample
- `regular_sample(batch_size)`: 70% regular + 30% random

### Class: `DQN`

**Purpose**: Simple Multi-Layer Perceptron for Q-values.

**Architecture**:
```
Linear(inputs → val) → ReLU
Linear(val → val) → ReLU
Linear(val → val) → ReLU
Linear(val → outputs) → log_softmax
```

### Class: `Agent`

**Purpose**: Base DQN agent class.

#### Key Methods

**`select_action(state)`**:
```python
eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    exp(-steps_done / epsilon_decay)

if random() > eps_threshold:
    return policy_net(state).max(1)[1]  # Exploit
else:
    return random_action  # Explore
```

**`optimize_model()`**:
Standard DQN optimization with Huber loss.

### Class: `MultiAgent`

**Purpose**: One DQN agent per node (decentralized).

#### Constructor
```python
def __init__(self, env):
    self.num_agents = env.observation_space.nvec[0]
    self.agents = []
    for node in nodes:
        num_outputs = len(neighbors(node))
        self.agents.append(
            Agent(num_outputs, DQN(num_inputs, num_outputs), ...)
        )
```

#### Method: `run(episodes)`

**Algorithm**:
```
For each episode:
    reset environment
    curr_agent = source_node_agent
    
    while not done:
        action = agents[curr_agent].select_action(state)
        next_state, reward, done = env.step(action)
        agents[curr_agent].memory.push(...)
        agents[curr_agent].optimize_model()
        curr_agent = next_node_agent
    
    # Update target networks periodically
```

### Constants
| Variable | Value | Description |
|----------|-------|-------------|
| `BATCH_SIZE` | `64*3 = 192` | Training batch |
| `GAMMA` | `0.999` | Discount factor |
| `EPS_START` | `1.0` | Initial exploration |
| `EPS_END` | `0.01` | Final exploration |
| `TARGET_UPDATE` | `1` | Target network sync frequency |

---

## 8. `models/GCN.py`

### Purpose
Graph Attention Network (GAT) model and agent for centralized routing.

### Class: `GCN`

**Architecture**:
```
Input: 3-dim features (note: different from GCN_env's 4-dim!)
  ↓
GATConv(3 → num_nodes//3) + SELU
  ↓
GATConv(num_nodes//3 → num_nodes//2) + SELU
  ↓
Flatten: (batch, num_nodes × num_nodes//2)
  ↓
Linear(flattened → num_nodes) + SELU
  ↓
Linear(num_nodes → out)
  ↓
Mask invalid actions (set to -inf)
  ↓
Output: Q-values
```

**Note**: The GCN class expects 3 input features, but GCN_env produces 4. This is a **discrepancy** in the RP15 codebase that the integrated module fixes.

### Class: `GCN_Agent`

**Purpose**: DQN agent using GAT model.

#### Constants (different from DQN.py!)
```python
TARGET_UPDATE = 14_000
EPS_DECAY = 1000
EPS_START = 0.95
BATCH_SIZE = 128
EPS_END = 0.001
GAMMA = 0.99
```

#### Method: `run(num_episodes)`

**Algorithm**:
```
For each episode:
    state, data = env.reset()
    mask = data['valid_actions']
    memory = ReplayMemory(3000)  # Fresh per episode!
    
    for step in count():
        action = select_action(state)
        next_state, reward, done = env.step(action)
        memory.push(...)
        optimize_model()
        
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            record_metrics()
            break
    
    scheduler.step()  # Learning rate decay
```

**Key Difference from Integrated**:
- Memory is **reset each episode** (`self.memory = ReplayMemory(3_000)`)
- Integrated uses persistent memory across episodes

---

## 9. `models/GraphSAGE.py`

### Purpose
GraphSAGE alternative to GAT for node embeddings.

### Class: `GraphSAGEPolicy`

**Architecture**:
```
Input: 4-dim node features
  ↓
SAGEConv(4 → num_nodes//3) + ReLU
  ↓
SAGEConv(num_nodes//3 → num_nodes//2) + ReLU
  ↓
Flatten: (batch, num_nodes × num_nodes//2)
  ↓
Linear(flattened → num_nodes) + ReLU
  ↓
Linear(num_nodes → out)
  ↓
Mask invalid actions
  ↓
Output: Q-values
```

**Forward Pass**:
```python
def forward(self, state):
    loader, mask = state
    
    for batch in loader:
        assert batch.x.shape[1] == 4  # Explicitly checks 4 features!
        x = self.conv1(batch.x, batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index)
        x = F.relu(x)
    
    x = x.split(self.num_nodes)
    x = x.flatten()
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    
    x[~mask] = float('-inf')
    return x
```

### Class: `GCN_Agent` (in GraphSAGE.py)

Same as `GCN.py`'s `GCN_Agent` but with different constants:
```python
EPS_DECAY = 17_000  # Slower decay than GCN (1000)
```

---

## 10. `helper/graph.py`

### Purpose
Graph manipulation and path computation utilities.

### Functions

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_neighbors(graph, node)` | `nx.Graph, int` | `list` | List of neighbor nodes |
| `compute_path_length(graph, path)` | `nx.Graph, list` | `float` | Sum of edge weights |
| `compute_flow_value(graph, path)` | `nx.Graph, list` | `float` | Minimum capacity (bottleneck) |
| `compute_best_flow(graph, src, tgt)` | `nx.Graph, int, int` | `float` | Maximum flow bottleneck |
| `get_max_neighbors(graph)` | `nx.Graph` | `int` | Maximum node degree |
| `draw_graph(graph)` | `nx.Graph` | None | Visualize graph |
| `adj_mat(graph)` | `nx.Graph` | `np.matrix` | Adjacency matrix |

### Function: `compute_path_length(graph, path)`

```python
def compute_path_length(graph, path):
    path_length = 0
    for i in range(len(path) - 1):
        path_length += graph[path[i]][path[i+1]]["weight"]
    return path_length
```

### Function: `compute_flow_value(graph, path)`

```python
def compute_flow_value(graph, path):
    worst_capacity = 1.0
    for i in range(len(path) - 1):
        edge_capacity = graph[path[i]][path[i+1]]["capacity"]
        worst_capacity = min(worst_capacity, edge_capacity)
    return worst_capacity
```

---

## Comparison: RP15 vs Integrated

| Aspect | RP15 | Integrated (DT-GCN) |
|--------|------|---------------------|
| Node Features | 4-dim | 7-dim |
| Env Class | `Env` | `HybridEnv` |
| Model Input | 3 or 4 features | 7 features |
| Reward | Base routing | Enhanced (jamming-aware) |
| Link Weights | Static BRITE | Dynamic (RadioMap-based) |
| Anomaly Detection | None | AnomalyBridge |
| Jammer Awareness | None | JammerDetector |
| Memory Reset | Per episode | Persistent |
| Dataset | Generated | Pre-loaded from RP12 |

---

## Known Discrepancies

1. **Feature Dimension Mismatch**:
   - `GCN_env.py`: Produces 4 features
   - `GCN.py`: Expects 3 features
   - `GraphSAGE.py`: Expects 4 features (with assertion)

2. **Memory Management**:
   - RP15: `self.memory = ReplayMemory(3_000)` inside episode loop
   - Integrated: Persistent `self.memory` across episodes

3. **Device Configuration**:
   - RP15: Hardcoded `"cpu"`
   - Integrated: Auto-detects CUDA

These discrepancies are resolved in the integrated module.
