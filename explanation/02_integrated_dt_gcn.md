# Integrated DT-GCN Module Deep Dive

## Module Overview

The `integrated_dt_gcn/` package is the **core integration layer** that combines Digital Twin signal analysis with GNN-based routing. It bridges the RP12_paper datasets with the RP15 routing system.

---

## File-by-File Analysis

---

## 1. `__init__.py`

### Purpose
Package initialization file that marks `integrated_dt_gcn/` as a Python package.

### Content
```python
# Integration: Digital Twin + RP15 Multi-Agent DQN with GCN
"""
Hybrid approach: 50nodes.brite topology + DT signal data
"""
```

### Dependencies
None - simple docstring.

---

## 2. `config.py`

### Purpose
Central configuration file containing **all global constants** used across the integrated system.

### Dependencies
- `torch`: For device detection (CUDA/CPU)

### Variables

| Variable | Type | Value | Description |
|----------|------|-------|-------------|
| `device` | `torch.device` | `cuda` or `cpu` | Computation device, auto-detected |
| `BRITE_FILE` | `str` | `"../RP15/50nodes.brite"` | Relative path to topology file |
| `DT_DATASET_DIR` | `str` | `"../RP12_paper/datasets"` | Path to DT datasets |
| `NUM_NODES` | `int` | `50` | Number of nodes in network |
| `NUM_EDGES` | `int` | `100` | Number of edges in network |
| `DT_MAP_SIZE` | `int` | `41` | Radio map grid size (41×41) |
| `BRITE_COORD_RANGE` | `int` | `100` | BRITE coordinate range (0-99) |
| `COORD_SCALE` | `float` | `0.4` | Scale factor: BRITE → DT coordinates |
| `NODE_FEATURE_DIM` | `int` | `7` | Node feature dimension (4 + 3 DT) |
| `RSS_CONNECTIVITY_THRESHOLD` | `float` | `-85` | Min RSS (dBm) for edge connectivity |
| `RSS_REFERENCE` | `float` | `-70` | Reference RSS for normalization |
| `LAMBDA_JAM` | `float` | `0.5` | Jamming penalty coefficient |
| `ALPHA_ANOMALY` | `float` | `0.3` | Anomaly dampening factor |
| `RESILIENCE_BONUS` | `float` | `0.3` | Bonus for avoiding jammed nodes |
| `EPS_DECAY` | `int` | `1000` | Epsilon decay rate |
| `EPS_START` | `float` | `0.95` | Initial exploration rate |
| `EPS_END` | `float` | `0.001` | Minimum exploration rate |
| `BATCH_SIZE` | `int` | `128` | Training batch size |
| `GAMMA` | `float` | `0.99` | Discount factor |
| `TARGET_UPDATE` | `int` | `14000` | Steps between target network updates |
| `LEARNING_RATE` | `float` | `0.001` | Adam optimizer learning rate |

### Coordinate Scaling

```
BRITE coordinates: (0-99, 0-99)
DT coordinates:    (0-40, 0-40)

Conversion: x_dt = x_brite × 0.4
```

This scaling maps the 50 mesh nodes (with positions in 100×100 space) to the 41×41 radio map grid.

---

## 3. `brite_loader.py`

### Purpose
Parse and load network topology from BRITE format files.

### Dependencies
- `networkx`: Graph representation
- `config.py`: DT_MAP_SIZE, BRITE_COORD_RANGE, COORD_SCALE

### Class: `BRITELoader`

#### Constructor
```python
def __init__(self, filepath: str)
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to `.brite` file |

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to BRITE file |
| `graph` | `nx.Graph` | Loaded NetworkX graph |

#### Method: `load_graph()`

**Purpose**: Parse BRITE file and create NetworkX graph.

**Input**: None (uses `self.filepath`)

**Output**: `nx.Graph` with:
- 50 nodes, each with `pos` attribute (x, y)
- 100 edges, each with `weight` (latency) and `capacity` (bandwidth)

**Algorithm**:
```
1. Read file lines
2. Parse header: "Topology: ( 50 Nodes, 100 Edges )"
3. Parse node section (lines 4-53):
   - Format: node_id \t x \t y
   - Add node with pos=(x, y)
4. Parse edge section (lines 57-156):
   - Format: edge_id \t src \t dst \t weight \t capacity
   - Add edge with weight and capacity
5. Return graph
```

#### Method: `scale_position_to_dt(pos)`

**Purpose**: Convert BRITE coordinates to DT radio map coordinates.

**Input**: `pos: Tuple[int, int]` - BRITE position (0-99)

**Output**: `Tuple[int, int]` - DT position (0-40)

**Formula**:
```python
x_dt = min(40, int(pos[0] * 0.4))
y_dt = min(40, int(pos[1] * 0.4))
```

#### Method: `get_scaled_positions()`

**Purpose**: Get all node positions in DT coordinates.

**Output**: `Dict[int, Tuple[int, int]]` - node_id → (x_dt, y_dt)

---

## 4. `dataset_loader.py`

### Purpose
Load and manage Digital Twin datasets from RP12_paper.

### Dependencies
- `pickle`: For loading .pkl files
- `numpy`: Array operations
- `pathlib.Path`: Path handling

### Class: `DTDatasetLoader`

#### Constructor
```python
def __init__(self, dataset_dir: str)
```

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_dir` | `Path` | Path to datasets directory |
| `_radio_maps` | `dict` | Cache: dataset_nr → RadioMap list |
| `_measurements` | `dict` | Cache: dataset_nr → MeasurementCollection |
| `_path_loss` | `dict` | Cache: dataset_nr → PathLossMapCollection |

#### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load_radio_maps(dataset_nr)` | `int` (0-5) | `List[RadioMap]` | Load 30,000 RadioMap objects |
| `load_measurements(dataset_nr)` | `int` | `MeasurementCollection` | Load measurement diffs |
| `load_path_loss(dataset_nr)` | `int` | `PathLossMapCollection` | Load path loss maps |
| `get_scenario_count(dataset_nr)` | `int` | `int` | Number of scenarios (30,000) |
| `get_jammed_indices(dataset_nr)` | `int` | `List[int]` | Indices with jammers |
| `get_normal_indices(dataset_nr)` | `int` | `List[int]` | Indices without jammers |
| `get_normal_measurements(dataset_nr)` | `int` | `List[np.ndarray]` | Diffs from normal scenarios |
| `get_radio_map_array(scenario_idx, dataset_nr)` | `int, int` | `np.ndarray (41,41)` | RSS values in dBm |
| `has_jammer(scenario_idx, dataset_nr)` | `int, int` | `bool` | Check jammer presence |
| `get_jammer_positions(scenario_idx, dataset_nr)` | `int, int` | `List[Tuple]` | Jammer (x, y) positions |
| `get_measurement_diff(scenario_idx, dataset_nr)` | `int, int` | `np.ndarray (25,)` | DT - actual RSS diffs |
| `get_measurement_points(dataset_nr)` | `int` | `Tuple[List, List]` | (meas_x, meas_y) coordinates |

#### Dataset Structure

```
Dataset Number → Noise Level (σ)
0 → σ = 0 dB (no noise)
1 → σ = 1 dB
2 → σ = 2 dB
...
5 → σ = 5 dB
```

Each dataset contains:
- **30,000 RadioMap objects**: 41×41 RSS grid, transmitter positions, jammer positions
- **30,000 measurement diffs**: 25-point arrays (DT prediction - actual measurement)
- ~50% scenarios with jammers, ~50% without

---

## 5. `digital_twin/mesh_twin.py`

### Purpose
Model the mesh network using Digital Twin radio maps to derive **link quality metrics**.

### Dependencies
- `numpy`: Numerical operations
- `networkx`: Graph structure
- `config.py`: RSS_REFERENCE

### Class: `MeshDigitalTwin`

#### Constructor
```python
def __init__(self, graph: nx.Graph, scaled_positions: Dict[int, Tuple[int, int]])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.Graph` | BRITE topology graph |
| `scaled_positions` | `dict` | Node positions in DT coords (0-40) |

#### Method: `update_link_weights(radio_map)`

**Purpose**: Update edge weights using RSS from radio map.

**Input**: `radio_map: np.ndarray (41, 41)` - RSS values in dBm

**Algorithm**:
```
For each edge (u, v):
  1. Get positions: pos_u, pos_v
  2. Get RSS at each endpoint from radio_map
  3. Compute average: avg_rss = (rss_u + rss_v) / 2
  4. Convert to latency: latency = _rss_to_latency(avg_rss)
  5. Convert to capacity: capacity = _rss_to_capacity(avg_rss)
  6. Update edge: graph[u][v]['weight'] = latency
                  graph[u][v]['capacity'] = capacity
```

#### Method: `_rss_to_latency(rss)`

**Purpose**: Convert RSS (dBm) to latency (0-1).

**Formula**:
```python
normalized = (rss - RSS_REFERENCE) / 30.0  # RSS_REFERENCE = -70
latency = 1.0 / (1.0 + exp(normalized))
return clamp(latency, 0.01, 0.99)
```

**Behavior**:
- Higher RSS → Lower latency (better link)
- RSS = -100 dBm → latency ≈ 1.0
- RSS = -50 dBm → latency ≈ 0.0

#### Method: `_rss_to_capacity(rss)`

**Purpose**: Convert RSS (dBm) to capacity (0-1).

**Formula**:
```python
capacity = (rss + 100) / 50.0
return clamp(capacity, 0.01, 1.0)
```

**Behavior**:
- Higher RSS → Higher capacity (better link)
- RSS = -100 dBm → capacity ≈ 0.0
- RSS = -50 dBm → capacity ≈ 1.0

---

## 6. `digital_twin/anomaly_bridge.py`

### Purpose
Convert DT measurement differences to **per-node anomaly scores** for jammer detection.

### Dependencies
- `numpy`: Statistical operations
- `config.py`: DT_MAP_SIZE

### Class: `AnomalyBridge`

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `threshold` | `float` | Anomaly threshold (mean + 2σ) |
| `mean` | `float` | Mean of normal diffs |
| `std` | `float` | Std of normal diffs |
| `is_trained` | `bool` | Training status flag |

#### Method: `train(normal_measurements)`

**Purpose**: Learn threshold from non-jammed scenarios.

**Input**: `List[np.ndarray]` - List of (25,) diff arrays from normal scenarios

**Algorithm**:
```
1. Concatenate all absolute diffs
2. Compute: mean = mean(|diffs|)
            std = std(|diffs|)
3. Set threshold = mean + 2 × std  (2-sigma rule)
4. is_trained = True
```

#### Method: `compute_measurement_scores(meas_diff)`

**Purpose**: Convert measurement diffs to anomaly scores.

**Input**: `np.ndarray (25,)` - DT - actual RSS differences

**Output**: `np.ndarray (25,)` - Anomaly scores (0-1)

**Formula**:
```python
abs_diff = |meas_diff|
score = 0  if abs_diff <= mean
score = min(1.0, (abs_diff - mean) / (2 × threshold))  if abs_diff > mean
```

#### Method: `map_to_nodes(meas_scores, meas_x, meas_y, node_positions)`

**Purpose**: Map 25 measurement point scores to 50 node scores.

**Algorithm**:
```
For each node:
  1. Find nearest measurement point (Euclidean distance)
  2. If distance < 1.0: node_score = meas_score[nearest]
  3. Else: node_score = meas_score[nearest] × exp(-dist/10)
```

**Rationale**: Anomaly scores decay with distance from measurement points.

### Class: `JammerDetector`

#### Constructor
```python
def __init__(self, jam_radius: float = 10.0)
```

#### Method: `get_jammed_nodes(jammer_positions, node_positions)`

**Purpose**: Binary detection - which nodes are jammed.

**Algorithm**:
```
For each node:
  For each jammer:
    If distance(node, jammer) <= jam_radius:
      node is jammed
```

#### Method: `get_jam_probabilities(jammer_positions, node_positions)`

**Purpose**: Soft detection - jamming probability per node.

**Formula**:
```python
if dist <= jam_radius:
    prob = 1.0
else:
    prob = exp(-(dist - jam_radius) / 5.0)
```

---

## 7. `environment/hybrid_env.py`

### Purpose
Gym environment that combines BRITE topology with DT signal data.

### Dependencies
- `gym`: RL environment base class
- `torch`, `torch_geometric`: Tensor operations
- `networkx`: Graph operations
- All `integrated_dt_gcn` submodules

### Class: `HybridEnv`

#### Constructor
```python
def __init__(self, brite_path: str, dt_loader: DTDatasetLoader,
             anomaly_bridge: AnomalyBridge, dataset_nr: int = 0)
```

#### Key Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.Graph` | Current network graph |
| `scaled_positions` | `dict` | Node positions in DT coords |
| `mesh_twin` | `MeshDigitalTwin` | Link weight updater |
| `anomaly_bridge` | `AnomalyBridge` | Anomaly scorer |
| `jammer_detector` | `JammerDetector` | Jammer proximity detector |
| `num_nodes` | `int` | Number of nodes (50) |
| `jammed_nodes` | `Set[int]` | Currently jammed node IDs |
| `anomaly_scores` | `Dict[int, float]` | Per-node anomaly scores |
| `jam_probabilities` | `Dict[int, float]` | Per-node jam probabilities |
| `edge_index` | `torch.Tensor` | PyG edge index (2, 200) |

#### Method: `reset()`

**Purpose**: Reset environment with new scenario from dataset.

**Output**: `Tuple[Data, Dict]` - (PyG Data object, info dict with valid_actions)

**Algorithm**:
```
1. Increment episode counter
2. Select scenario (50% jammed, 50% normal)
3. Load radio map for scenario
4. Update link weights via MeshDigitalTwin
5. Get jammer positions, detect jammed nodes
6. Compute anomaly scores via AnomalyBridge
7. Select random source-target pair with valid path
8. Compute 7-dimensional node features
9. Return (Data(x=features, edge_index), {'valid_actions': mask})
```

#### Method: `step(action)`

**Purpose**: Execute routing action.

**Input**: `action: int` - Node ID to move to

**Output**: `Tuple[Data, float, bool, Dict]` - (observation, reward, done, info)

**Algorithm**:
```
1. Validate action (must be neighbor of current node)
2. If invalid: return penalty reward -1.0
3. Add node to path
4. Track if stepped on jammed node
5. Compute enhanced reward via compute_enhanced_reward()
6. Compute new node features
7. Return new observation
```

#### Method: `_compute_node_features()`

**Purpose**: Compute 7-dimensional feature vector for each node.

**Output**: `torch.Tensor (50, 7)` on device

**Features**:
```python
x[node, 0] = 1.0 if node == current_node else 0.0    # is_source
x[node, 1] = 1.0 if node == target else 0.0           # is_dest
x[node, 2] = mean(edge_weight for neighbors)          # avg_latency
x[node, 3] = mean(edge_capacity for neighbors)        # avg_bandwidth
x[node, 4] = anomaly_scores.get(node, 0.0)            # anomaly_score
x[node, 5] = jam_probabilities.get(node, 0.0)         # jam_probability
x[node, 6] = mean(jam_prob for n in neighbors)        # neighbor_jam_avg
```

---

## 8. `environment/enhanced_reward.py`

### Purpose
Compute jamming-aware rewards for routing.

### Dependencies
- `networkx`: Path operations
- `config.py`: LAMBDA_JAM, ALPHA_ANOMALY, RESILIENCE_BONUS

### Function: `compute_base_reward(graph, target, path)`

**Purpose**: Standard routing reward (from RP15).

**Output**: `Tuple[float, bool]` - (reward, done)

**Reward Structure**:
```
+1.01: Reached target via optimal hop
-1.51: Reached target via suboptimal hop
(d_old - d_new): Progress toward target (positive = good)
-1.0: Moved away from target or timeout
```

### Function: `compute_enhanced_reward(...)`

**Purpose**: Full reward with DT integration.

**Formula**:
```python
R_total = R_base × (1 - α × anomaly) - λ × jammed + resilience_bonus

Where:
  R_base = base routing reward
  α = ALPHA_ANOMALY = 0.3
  anomaly = anomaly_scores[current_node]
  λ = LAMBDA_JAM = 0.5
  jammed = 1 if stepped on jammed node else 0
  resilience_bonus = 0.3 if (reached target AND avoided all jammed nodes)
```

---

## 9. `models/gcn_dt_aware.py`

### Purpose
GCN model and DQN agent with Digital Twin awareness.

### Dependencies
- `torch`, `torch.nn`: Neural network components
- `torch_geometric.nn.GATConv`: Graph Attention layer
- `torch_geometric.loader.DataLoader`: Batch loading
- `config.py`: All training parameters

### Class: `ReplayMemory`

**Purpose**: Experience replay buffer.

| Method | Description |
|--------|-------------|
| `push(*args)` | Store transition (state, action, next_state, reward) |
| `sample(batch_size)` | Random sample of transitions |

**Capacity**: 5000 transitions

### Class: `GCN_DTAware`

**Purpose**: Graph neural network for Q-value prediction.

#### Architecture
```
Input: 7-dim node features
  ↓
GATConv(7 → 16) + SELU
  ↓
GATConv(16 → 25) + SELU
  ↓
Flatten: (batch, 50×25) = (batch, 1250)
  ↓
Linear(1250 → 50) + SELU
  ↓
Linear(50 → 50)
  ↓
Mask invalid actions
  ↓
Output: Q-values for 50 nodes
```

#### Forward Pass
```python
def forward(self, state):
    loader, mask = state
    
    for batch in loader:
        out = conv1(batch.x, batch.edge_index)  # GATConv
        out = selu(out)
        out = conv2(out, batch.edge_index)      # GATConv
    
    out = out.split(num_nodes)  # Reshape
    out = out.flatten(start_dim=1)
    out = selu(fc1(out))
    out = fc2(out)
    
    out[~mask] = float('-inf')  # Mask invalid actions
    return out
```

### Class: `GCN_DTAgent`

**Purpose**: DQN agent using GCN_DTAware model.

#### Constructor
```python
def __init__(self, num_nodes, policy_net, target_net, env)
```

#### Key Components
- **policy_net**: Current Q-network (trained)
- **target_net**: Stable Q-network (periodic updates)
- **memory**: ReplayMemory(5000)
- **optimizer**: Adam(lr=0.001)

#### Metrics Tracked
```python
self.metrics = {
    'loss': [],           # Per-step loss
    'reward': [],         # Per-step reward
    'path_length': [],    # Episode lengths
    'eps_reward': [],     # Episode total rewards
    'jammed_steps': [],   # Jammed steps per episode
    'step_latency': [],   # Per-step latency
    'step_bandwidth': [], # Per-step bandwidth
    'eps_total_latency': [],  # Episode total latency
    'eps_avg_bandwidth': [],  # Episode avg bandwidth
    'eps_avg_pdr': []         # Episode avg PDR
}
```

#### Method: `select_action(state, valid_actions)`

**Algorithm** (ε-greedy):
```python
eps = EPS_END + (EPS_START - EPS_END) × exp(-steps_done / EPS_DECAY)

if random() > eps:
    # Exploit: use policy network
    q_values = policy_net(state)
    action = argmax(q_values)
else:
    # Explore: random valid action
    action = random_choice(valid_actions)
```

#### Method: `optimize_model()`

**Algorithm** (DQN with experience replay):
```
1. Sample batch from replay memory
2. Compute Q(s, a) from policy_net
3. Compute V(s') from target_net
4. Compute expected Q: E[Q] = reward + γ × V(s')
5. Compute Huber loss: L = smooth_l1_loss(Q, E[Q])
6. Backpropagate and clip gradients to [-1, 1]
7. Update policy_net weights
```

#### Method: `run(num_episodes)`

**Training Loop**:
```
For each episode:
    reset environment
    For each step:
        select action (ε-greedy)
        execute action
        store transition in memory
        optimize every 4 steps
        update target network every 14000 steps
        if done: record metrics, break
```

---

## Summary Table

| File | Primary Class | Key Function |
|------|--------------|--------------|
| `config.py` | N/A | Global constants |
| `brite_loader.py` | `BRITELoader` | Parse network topology |
| `dataset_loader.py` | `DTDatasetLoader` | Load DT datasets |
| `mesh_twin.py` | `MeshDigitalTwin` | RSS → Link weights |
| `anomaly_bridge.py` | `AnomalyBridge` | Measurement diffs → Anomaly scores |
| `hybrid_env.py` | `HybridEnv` | 7-dim features + Enhanced rewards |
| `enhanced_reward.py` | N/A | Jamming-aware reward function |
| `gcn_dt_aware.py` | `GCN_DTAware`, `GCN_DTAgent` | Model + Training |
