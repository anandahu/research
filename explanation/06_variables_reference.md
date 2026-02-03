# Variables and Constants Reference

## Quick Reference Guide

This document provides a comprehensive reference for all variables, constants, and configuration parameters used across the codebase.

---

## 1. Global Configuration (`integrated_dt_gcn/config.py`)

### Device Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `device` | Auto-detect | `"cuda"` if GPU available, else `"cpu"` |

### Path Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `BRITE_FILE` | `"../RP15/50nodes.brite"` | Network topology file |
| `DT_DATASET_DIR` | `"../RP12_paper/datasets"` | DT datasets directory |

### Network Parameters

| Variable | Value | Type | Description |
|----------|-------|------|-------------|
| `NUM_NODES` | `50` | `int` | Number of nodes in mesh network |
| `NUM_EDGES` | `100` | `int` | Number of edges in network |
| `DT_MAP_SIZE` | `41` | `int` | Radio map grid dimension (41×41) |
| `BRITE_COORD_RANGE` | `100` | `int` | BRITE coordinate range (0-99) |
| `COORD_SCALE` | `0.4` | `float` | Scale: BRITE → DT coords |
| `NODE_FEATURE_DIM` | `7` | `int` | Features per node (RP15: 4) |

### RSS/Signal Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `RSS_CONNECTIVITY_THRESHOLD` | `-85` | dBm | Min RSS for edge existence |
| `RSS_REFERENCE` | `-70` | dBm | Reference RSS for normalization |

### Reward Coefficients

| Variable | Value | Description |
|----------|-------|-------------|
| `LAMBDA_JAM` | `0.5` | Penalty for stepping on jammed node |
| `ALPHA_ANOMALY` | `0.3` | Dampening factor for anomaly |
| `RESILIENCE_BONUS` | `0.3` | Bonus for avoiding all jammed nodes |

### Training Hyperparameters

| Variable | Value | Description |
|----------|-------|-------------|
| `BATCH_SIZE` | `128` | Training batch size |
| `GAMMA` | `0.99` | Discount factor (γ) |
| `EPS_START` | `0.95` | Initial exploration rate |
| `EPS_END` | `0.001` | Minimum exploration rate |
| `EPS_DECAY` | `1000` | Exploration decay rate |
| `TARGET_UPDATE` | `14000` | Steps between target net updates |
| `LEARNING_RATE` | `0.001` | Adam optimizer learning rate |

---

## 2. RP15 Constants

### `RP15/config.py`

| Variable | Value | Description |
|----------|-------|-------------|
| `device` | `"cpu"` | Hardcoded CPU (no auto-detect) |

### `RP15/models/DQN.py`

| Variable | Value | Description |
|----------|-------|-------------|
| `BATCH_SIZE` | `64*3 = 192` | Larger than integrated |
| `GAMMA` | `0.999` | Higher discount factor |
| `EPS_START` | `1.0` | Full exploration initially |
| `EPS_END` | `0.01` | Higher minimum than integrated |
| `TARGET_UPDATE` | `1` | Very frequent updates |

### `RP15/models/GCN.py`

| Variable | Value | Description |
|----------|-------|-------------|
| `TARGET_UPDATE` | `14_000` | Same as integrated |
| `EPS_DECAY` | `1000` | Same as integrated |
| `EPS_START` | `0.95` | Same as integrated |
| `BATCH_SIZE` | `128` | Same as integrated |
| `EPS_END` | `0.001` | Same as integrated |
| `GAMMA` | `0.99` | Same as integrated |

### `RP15/models/GraphSAGE.py`

| Variable | Value | Description |
|----------|-------|-------------|
| `EPS_DECAY` | `17_000` | Slower decay for GraphSAGE |
| Others | Same as GCN.py | |

### `RP15/train.py`

| Variable | Value | Description |
|----------|-------|-------------|
| `lwr` | `-1.51` | "Lowest winning reward" threshold |
| `NODE_FEATURE_DIM` | `4` | RP15 baseline features |

---

## 3. RP12 Paper Constants

### Dataset Generation Configuration

**`pathloss_map_generation.py`**:

| Variable | Value | Description |
|----------|-------|-------------|
| `f_c` | `2.4e9` Hz | Carrier frequency (2.4 GHz) |
| `scene_size` | `[41, 41]` | Scene dimensions (meters) |
| `resolution` | `1` | 1 meter per pixel |
| `noise_std` | 0-5 dB | Shadowing std (per dataset) |
| `d_corr` | `1` | Correlation distance (m) |

**`radio_map_generation.py`**:

| Variable | Value | Description |
|----------|-------|-------------|
| `tx_power` | `20` dBm | Transmitter power |
| `range_num_tx` | `[10, 10]` | Always 10 transmitters |
| `range_jam_power` | `[20, 20]` dBm | Jammer power |
| `range_num_jam` | `[0, 1]` | 0 or 1 jammer |
| `num_radiomaps` | `30000` | Scenarios per dataset |

**`measurement_generation.py`**:

| Variable | Value | Description |
|----------|-------|-------------|
| `measurement_method` | `'grid'` | Grid-based sampling |
| `grid_size` | `8` | 8-meter spacing |
| `pos_std` | `0` | TX position uncertainty |

---

## 4. Node Feature Dimensions

### RP15 Baseline (4-dim)

| Index | Feature | Range | Source |
|-------|---------|-------|--------|
| 0 | is_source | {0, 1} | Environment state |
| 1 | is_dest | {0, 1} | Environment state |
| 2 | avg_latency | [0, 1] | BRITE edge weights |
| 3 | avg_bandwidth | [0, 1] | BRITE edge capacities |

### Integrated (7-dim)

| Index | Feature | Range | Source |
|-------|---------|-------|--------|
| 0 | is_source | {0, 1} | Environment state |
| 1 | is_dest | {0, 1} | Environment state |
| 2 | avg_latency | [0, 1] | DT-updated edge weights |
| 3 | avg_bandwidth | [0, 1] | DT-updated edge capacities |
| 4 | anomaly_score | [0, 1] | AnomalyBridge |
| 5 | jam_probability | [0, 1] | JammerDetector |
| 6 | neighbor_jam_avg | [0, 1] | Mean of neighbors' jam_prob |

---

## 5. Reward Values

### Base Rewards (RP15)

| Condition | Reward | Description |
|-----------|--------|-------------|
| Target reached, optimal hop | `+1.01` | Best possible outcome |
| Target reached, suboptimal hop | `-1.51` | Reached but not optimal |
| Progress toward target | `+(d_old - d_new)` | Positive distance reduction |
| Moved away from target | `-1.0` | Negative progress |
| Max steps exceeded | `-1.0` | Timeout penalty |
| Invalid action | `-1.0` | Tried non-neighbor |

### Enhanced Rewards (Integrated)

| Component | Formula | Description |
|-----------|---------|-------------|
| Dampened base | `R_base × (1 - 0.3 × anomaly)` | Reduce reward on anomalous nodes |
| Jam penalty | `-0.5` if jammed | Penalty for visiting jammed |
| Resilience bonus | `+0.3` | Target reached, no jammed nodes hit |

**Full Formula**:
```
R_total = R_base × (1 - ALPHA_ANOMALY × anomaly) 
          - LAMBDA_JAM × jammed 
          + resilience_bonus
```

---

## 6. Model Architecture Parameters

### GCN_DTAware

| Layer | Input | Output | Activation |
|-------|-------|--------|------------|
| GATConv 1 | 7 | 16 | SELU |
| GATConv 2 | 16 | 25 | SELU |
| Flatten | 50×25 | 1250 | - |
| Linear 1 | 1250 | 50 | SELU |
| Linear 2 | 50 | 50 | - |
| Action mask | 50 | 50 | -inf for invalid |

### GCN (RP15 Baseline)

| Layer | Input | Output | Activation |
|-------|-------|--------|------------|
| GATConv 1 | 3 | 16 | SELU |
| GATConv 2 | 16 | 25 | SELU |
| Flatten | 50×25 | 1250 | - |
| Linear 1 | 1250 | 50 | SELU |
| Linear 2 | 50 | 50 | - |

### GraphSAGE

| Layer | Input | Output | Activation |
|-------|-------|--------|------------|
| SAGEConv 1 | 4 | 16 | ReLU |
| SAGEConv 2 | 16 | 25 | ReLU |
| Flatten | 50×25 | 1250 | - |
| Linear 1 | 1250 | 50 | ReLU |
| Linear 2 | 50 | 50 | - |

---

## 7. Anomaly Detection Parameters

### AnomalyBridge

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `mean + 2×std` | 2-sigma rule |
| `decay_factor` | `10.0` | Distance decay for node mapping |

### JammerDetector

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jam_radius` | `10.0` | DT units (~10 meters) |
| `decay_distance` | `5.0` | Soft probability decay |

---

## 8. Dataset Statistics

### RadioMap Datasets

| Dataset | Scenarios | Jammed | Normal | Size (MB) |
|---------|-----------|--------|--------|-----------|
| 0 | 30,000 | ~15,000 | ~15,000 | ~413 |
| 1-5 | 30,000 | ~15,000 | ~15,000 | ~413 each |

### Measurement Points

| Parameter | Value |
|-----------|-------|
| Grid size | 8 meters |
| Map size | 41×41 meters |
| Points per scenario | 25 (5×5 grid) |

---

## 9. Training Default Values

### train_integrated.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | `500` | Training episodes |
| `--dataset_nr` | `0` | Dataset number |
| `--normal_samples` | `2000` | Anomaly training samples |
| `--save` | `False` | Save model |
| `--plot` | `False` | Show plots |

### compare_models.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | `300` | Episodes per model |
| `--dataset_nr` | `0` | Dataset number |
| `--plot` | `True` | Generate plots |

---

## 10. Epsilon Decay Schedule

The exploration rate decays exponentially:

```python
eps = EPS_END + (EPS_START - EPS_END) × exp(-steps_done / EPS_DECAY)
```

| Steps | Epsilon (EPS_DECAY=1000) |
|-------|-------------------------|
| 0 | 0.95 |
| 500 | 0.58 |
| 1000 | 0.35 |
| 2000 | 0.13 |
| 3000 | 0.05 |
| 5000 | 0.01 |
| 10000 | 0.001 |
