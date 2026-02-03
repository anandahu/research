# System Architecture Overview

## 1. Introduction

This repository implements a **Digital Twin-Enhanced GNN-Based Routing System** for wireless mesh networks. The system combines:

1. **RP15 (Multi-Agent DQN + GNN Routing)**: Baseline routing using Graph Convolutional Networks
2. **RP12_paper (Digital Twin for Anomaly Detection)**: Radio environment simulation and jammer detection
3. **integrated_dt_gcn (Integrated System)**: Novel integration combining DT signals with GNN routing

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐      ┌────────────────┐      ┌──────────────────┐      │
│    │  RP12_paper  │      │ integrated_dt_ │      │      RP15        │      │
│    │   datasets/  │─────▶│      gcn/      │◀─────│  50nodes.brite   │      │
│    │              │      │                │      │                  │      │
│    │ • Radio Maps │      │ • HybridEnv    │      │ • Network        │      │
│    │ • Path Loss  │      │ • AnomalyBridge│      │   Topology       │      │
│    │ • Measurements│     │ • MeshDigitalTwin │   │ • Edge Weights   │      │
│    └──────────────┘      │ • GCN_DTAware  │      └──────────────────┘      │
│                          └───────┬────────┘                                │
│                                  │                                         │
│                                  ▼                                         │
│    ┌─────────────────────────────────────────────────────────────────┐    │
│    │                    train_integrated.py                          │    │
│    │                    compare_models.py                            │    │
│    └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Dataset Loading Pipeline

```
RP12_paper/datasets/
    │
    ├── fspl_RMdataset{N}.pkl ──────┐
    │   (30,000 RadioMap objects)   │
    │   41×41 RSS grid per scenario │
    │                               ▼
    ├── fspl_PLdataset{N}.pkl ── DTDatasetLoader ──▶ HybridEnv
    │   (PathLossMapCollection)     │
    │                               │
    └── fspl_measurements{N}.pkl ───┘
        (MeasurementCollection)
        25-point diffs per scenario
```

### 3.2 Feature Construction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     7-DIMENSIONAL NODE FEATURES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  From BRITE Topology (RP15):        From Digital Twin (RP12_paper):     │
│  ┌─────────────────────┐            ┌─────────────────────────────┐    │
│  │ [0] is_source       │            │ [4] anomaly_score           │    │
│  │ [1] is_destination  │            │     (from measurement diff) │    │
│  │ [2] avg_latency     │            │ [5] jam_probability         │    │
│  │ [3] avg_bandwidth   │            │     (distance to jammer)    │    │
│  └─────────────────────┘            │ [6] neighbor_jam_avg        │    │
│                                     │     (avg jam_prob of nbrs)  │    │
│                                     └─────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```
research/
├── integrated_dt_gcn/          # Core integration layer
│   ├── __init__.py             # Package init
│   ├── config.py               # Global configuration constants
│   ├── brite_loader.py         # BRITE topology parser
│   ├── dataset_loader.py       # DT dataset loader utility
│   ├── digital_twin/           # DT components
│   │   ├── mesh_twin.py        # Link quality from radio maps
│   │   └── anomaly_bridge.py   # Measurement diff → anomaly scores
│   ├── environment/            # RL environment
│   │   ├── hybrid_env.py       # Gym environment with 7-dim features
│   │   └── enhanced_reward.py  # Jamming-aware reward function
│   ├── models/                 # Neural network models
│   │   └── gcn_dt_aware.py     # GATConv model + DQN agent
│   └── results/                # Training outputs (CSV, PNG)
│
├── RP15/                       # Baseline GNN routing module
│   ├── 50nodes.brite           # Network topology file
│   ├── environment/
│   │   ├── GCN_env.py          # Baseline environment (4-dim features)
│   │   ├── env.py              # Link-hop environment
│   │   └── util.py             # Reward functions, graph utilities
│   ├── models/
│   │   ├── GCN.py              # GATConv baseline model
│   │   ├── GraphSAGE.py        # SAGEConv alternative
│   │   └── DQN.py              # Base agent classes
│   ├── helper/
│   │   └── graph.py            # Graph utility functions
│   └── train.py                # Standalone RP15 training
│
├── RP12_paper/                 # Digital Twin & dataset generation
│   ├── datasets/               # Pre-generated datasets (.pkl files)
│   ├── src/
│   │   ├── dataset_generation/
│   │   │   ├── pathloss_map_generation.py
│   │   │   ├── radio_map_generation.py
│   │   │   └── measurement_generation.py
│   │   ├── anomaly_detection/
│   │   │   ├── fspl_anomaly_detection.py
│   │   │   ├── supervised_detection.py
│   │   │   └── unsupervised_detection.py
│   │   └── utils/
│   │       ├── pl_utils.py     # PathLossMap classes
│   │       ├── radiomap_utils.py # RadioMap, Transmitter classes
│   │       └── description_file_utils.py
│   └── figures/                # Output visualizations
│
├── train_integrated.py         # Main training script
├── compare_models.py           # GCN vs GCN+DT comparison
├── plot_comparison_graph.py    # Visualization utilities
├── plot_training_graph.py      # Training metrics plots
└── plot_network_topology.py    # Network visualization
```

---

## 5. Execution Flow

### 5.1 Training Flow (`train_integrated.py`)

```
1. Load DT Datasets
   └── DTDatasetLoader(dataset_dir)
       ├── load_radio_maps(dataset_nr)      # 30,000 scenarios
       ├── get_jammed_indices()             # Filter jammed scenarios
       └── get_normal_measurements()        # For anomaly training

2. Train Anomaly Detector
   └── AnomalyBridge.train(normal_measurements)
       └── Compute mean, std, threshold (2-sigma rule)

3. Create Hybrid Environment
   └── HybridEnv(brite_path, loader, anomaly_bridge)
       ├── BRITELoader.load_graph()         # 50 nodes, 100 edges
       ├── MeshDigitalTwin(graph, positions)
       └── JammerDetector(jam_radius=10.0)

4. Create GCN Model
   └── GCN_DTAware(num_nodes=50, out_dim=50, node_feat_dim=7)
       ├── GATConv(7 → 16)
       ├── GATConv(16 → 25)
       ├── Linear(50*25 → 50)
       └── Linear(50 → 50)

5. Create Agent & Train
   └── GCN_DTAgent.run(num_episodes)
       ├── Environment reset (select scenario, compute features)
       ├── Action selection (ε-greedy with Q-network)
       ├── Step execution (compute enhanced reward)
       └── Experience replay optimization (Huber loss)

6. Save Results
   ├── episode_metrics_{timestamp}.csv
   ├── step_metrics_{timestamp}.csv
   ├── training_summary.csv
   └── training_metrics.png
```

### 5.2 Comparison Flow (`compare_models.py`)

```
1. Train GCN + DT (7 features)
   └── Full HybridEnv with all DT features

2. Train GCN Baseline (4 features)
   └── Same environment but zero out features [4:7]

3. Compare Results
   ├── Average Reward
   ├── Success Rate
   ├── Average Path Length
   ├── Average Jammed Steps
   └── Training Time
```

---

## 6. Key Innovations

### 6.1 Digital Twin Integration

The DT provides **two types of information**:

| Source | Information | Used For |
|--------|-------------|----------|
| Radio Maps | RSS at node positions | Link weight updates (latency, capacity) |
| Measurements | Diff between DT and actual | Anomaly detection (jammer presence) |

### 6.2 Enhanced Node Features

Traditional GNN routing uses 2-4 features. This system uses **7 features**:

| Index | Feature | Source | Range |
|-------|---------|--------|-------|
| 0 | is_source | RP15 | 0 or 1 |
| 1 | is_destination | RP15 | 0 or 1 |
| 2 | avg_latency | BRITE + RadioMap | [0, 1] |
| 3 | avg_bandwidth | BRITE + RadioMap | [0, 1] |
| 4 | anomaly_score | AnomalyBridge | [0, 1] |
| 5 | jam_probability | JammerDetector | [0, 1] |
| 6 | neighbor_jam_avg | JammerDetector | [0, 1] |

### 6.3 Enhanced Reward Function

```
R_total = R_base × (1 - α × anomaly) - λ × jammed + resilience_bonus

Where:
- R_base: Standard routing reward (progress toward target)
- α = 0.3: Anomaly dampening factor (ALPHA_ANOMALY)
- λ = 0.5: Jamming penalty coefficient (LAMBDA_JAM)
- resilience_bonus = 0.3: Bonus for avoiding jammed nodes (RESILIENCE_BONUS)
```

---

## 7. Component Interactions

```
                    ┌─────────────────┐
                    │ DTDatasetLoader │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌──────────────────┐
    │ MeshDigitalTwin │          │  AnomalyBridge   │
    │                 │          │                  │
    │ • RSS → Latency │          │ • Diff → Score   │
    │ • RSS → Capacity│          │ • Score → Nodes  │
    └────────┬────────┘          └────────┬─────────┘
             │                            │
             └──────────┬─────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │   HybridEnv    │
               │                │
               │ • 7-dim feats  │
               │ • Enhanced R   │
               └───────┬────────┘
                       │
                       ▼
              ┌────────────────┐
              │  GCN_DTAware   │
              │                │
              │ • GATConv x2   │
              │ • Q-values     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  GCN_DTAgent   │
              │                │
              │ • ε-greedy     │
              │ • Replay       │
              │ • Training     │
              └────────────────┘
```

---

## 8. Quick Start

### Training

```bash
# Train with Digital Twin integration (500 episodes)
python train_integrated.py --episodes 500 --save --plot

# Compare GCN vs GCN+DT
python compare_models.py --episodes 300 --plot
```

### Configuration

All key parameters are in `integrated_dt_gcn/config.py`:

```python
NUM_NODES = 50            # Network size
NODE_FEATURE_DIM = 7      # Features per node
LAMBDA_JAM = 0.5          # Jamming penalty
ALPHA_ANOMALY = 0.3       # Anomaly dampening
BATCH_SIZE = 128          # Training batch size
GAMMA = 0.99              # Discount factor
```
