# Paper vs. Implementation Comparison Report

**Project:** Resilient Routing in Wireless Mesh Networks using Digital Twin and GNN  
**Date:** March 2026  
**Source Papers:**  
- *Resilient Routing in Wireless Mesh Networks using Digital Twin and GNN.docx* (Main paper)  
- *Related works1.docx* (Related works + Experimental results)  

**Codebase:** `integrated_dt_gcn/`

---

## 1. Overview

This report compares the two submitted paper documents against the actual code implementation in `integrated_dt_gcn/`. The paper proposes a **Multi-Agent DT-GCN** routing framework that integrates a Digital Twin (DT), Graph Attention Networks (GAT), and Multi-Agent Deep Q-Networks (DQN) for resilient routing in a 50-node Wireless Mesh Network.

---

## 2. Core Claims vs. Implementation: Tabular Comparison

### 2.1 Architecture & Model Design

| Aspect | Paper Claims | Project Implementation | ✅/⚠️/❌ |
|---|---|---|---|
| **Network size** | 50-node Wireless Mesh Network | `NUM_NODES = 50`, BRITE file `50nodes.brite` | ✅ Match |
| **Number of edges** | Not explicitly stated | `NUM_EDGES = 100` in `config.py` | ✅ Consistent |
| **GNN type** | "Graph Convolutional Network (GCN)" in title, but "two Graph Attention Network layers" in methods | `GATConv` from PyG used in `gat_dt_aware.py` | ✅ Code uses GAT correctly |
| **GNN layers** | Two GAT layers + FC layers | `conv1 = GATConv(...)`, `conv2 = GATConv(...)`, `fc1`, `fc2` | ✅ Match |
| **Class naming** | Paper uses "GCN" in title, "GAT" in methods | Class renamed to `GAT_DTAware` in `gat_dt_aware.py` | ✅ Fixed (was GCN_DTAware) |
| **Node feature dimensions** | 7-dimensional feature vector | `NODE_FEATURE_DIM = 7` in `config.py` | ✅ Match |
| **Node features** | avg_latency, avg_bandwidth, anomaly_score, jam_probability, neighbor_jam_avg + src/dest flags | `[is_source, is_dest, avg_latency, avg_bandwidth, anomaly_score, jam_prob, neighbor_jam_avg]` in `hybrid_env.py` | ✅ Match |
| **Baseline model** | GCN Baseline with 4 features (no DT, no anomaly) | `compare_models.py` uses same `GAT_DTAware` with DT features zeroed out | ✅ Match |
| **Multi-Agent setup** | 50 agents (one per node) | `MultiAgentCoordinator` with 50 `NodeAgent` instances in `multi_agent.py` | ✅ Match |

---

### 2.2 Digital Twin (DT) Component

| Aspect | Paper Claims | Project Implementation | ✅/⚠️/❌ |
|---|---|---|---|
| **DT concept** | "Real-time virtual mirror of the real network" | `MeshDigitalTwin` class in `digital_twin/mesh_twin.py` | ✅ Match |
| **Radio Map** | "Environmental radio maps containing RSS measurements" | `DT_MAP_SIZE = 41`, `radio_map = (41,41)` array in `mesh_twin.py` | ✅ Match |
| **RSS → Latency** | "Sigmoidal transformation functions are used to convert RSS values into estimates" | `_rss_to_latency()` uses sigmoid: `1.0 / (1.0 + exp(normalized))` | ✅ Match |
| **RSS → Capacity** | RSS converted to capacity estimate | `_rss_to_capacity()` uses linear normalization: `(rss + 100) / 50.0` | ✅ Match |
| **Anomaly Detection** | "`AnomalyBridge` and `JammerDetector` modules" | Both classes present in `anomaly_bridge.py` | ✅ Match |
| **Anomaly threshold** | Not specified in paper (implied statistical) | `threshold = mean + 2 * std` (2-sigma) in `AnomalyBridge.train()` | ✅ Match |
| **Jammed node detection** | "Detection of anomalies and analysis of predictive failures" | `JammerDetector(jam_radius=10.0)` with distance-based detection | ✅ Match |
| **DT synchronization** | "Virtual representation is continuously updated" | `mesh_twin.update_link_weights(radio_map)` called on each `reset()` | ✅ Match |
| **Feature enrichment** | "Each node has a seven-dimensional feature vector" | `_compute_node_features()` builds all 7 features per node | ✅ Match |

---

### 2.3 Reward Function

| Aspect | Paper Claims | Paper Formula | Project Implementation | ✅/⚠️/❌ |
|---|---|---|---|---|
| **Base reward** | "Calculated according to the agent's progress towards the destination" | Distance progress reward | `compute_base_reward()` using `nx.astar_path_length` progress | ✅ Match |
| **Less-hop more-reward** | "It promotes less-hop more-reward policy" | Implicit | Progress reward gives higher reward for larger distance reduction | ✅ Match |
| **Anomaly dampening** | "Anomaly dampening factor" | `R_base × (1 - α × anomaly)` | `dampened_reward = base_reward * (1 - ALPHA_ANOMALY * anomaly)`, `ALPHA_ANOMALY = 0.3` | ✅ Match |
| **Jamming penalty** | "Negative reward when the agent selects a jammed node" | `−λ` if jammed | `jam_penalty = -LAMBDA_JAM` if `stepped_on_jammed`, `LAMBDA_JAM = 0.5` | ✅ Match |
| **Resilience bonus** | "Additional reward if routing completed successfully" | `+RESILIENCE_BONUS` | `RESILIENCE_BONUS = 0.3` added if destination reached without jammed nodes | ✅ Match |
| **Optimal vs. suboptimal** | Implied through reward structure | `+1.01` / `−1.51` | `return 1.01, True` / `return -1.51, True` in `compute_base_reward()` | ✅ Match |
| **Timeout penalty** | Implied (episode must terminate) | — | `if len(path) > 10 * len(graph.nodes): return -1.0, True` | ✅ Match |

---

### 2.4 DQN Training Setup

| Aspect | Paper Claims | Project Implementation | ✅/⚠️/❌ |
|---|---|---|---|
| **Algorithm** | Deep Q-Network (DQN) | DQN with policy + target networks | ✅ Match |
| **Experience replay** | "Interaction tuples are stored in a replay memory" | `ReplayMemory` class with `push()` and `sample()` | ✅ Match |
| **Exploration** | "Epsilon-greedy policy... exploration rate decays gradually" | `eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-steps / EPS_DECAY)` | ✅ Match |
| **Target network** | Implied by DQN architecture | `target_net` synced periodically | ✅ Match |
| **Loss function** | Implied by DQN | `F.smooth_l1_loss(...)` (= Huber loss) | ✅ Match |
| **Gradient clipping** | Not mentioned | `torch.nn.utils.clip_grad_norm_(parameters, 1.0)` — extra stability | ✅ Better than paper |
| **Multi-agent replay** | "Shared experience replay mechanism" | Shared `SharedReplayMemory` pool (`SHARED_POOL_SIZE = 50000`) in `multi_agent.py` | ✅ Match |
| **Training episodes** | 3,000 episodes | `num_episodes` configurable, default 500 (paper claims 3000) | ⚠️ Default mismatch |
| **Training time benchmarking** | 34.0 min (MA) vs 122.1 min (baseline) | `time.time()` wrappers in both `train_integrated.py` and `train_multi_agent.py` | ✅ Match |
| **Reproducibility** | Not discussed in paper | `--seed` arg with `random.seed`, `np.random.seed`, `torch.manual_seed` | ✅ Better than paper |

---

### 2.5 Multi-Agent Configuration

| Aspect | Paper Claims | Project Implementation | ✅/⚠️/❌ |
|---|---|---|---|
| **Number of agents** | 50 (one per node) | `MultiAgentCoordinator` creates `NodeAgent` for each graph node | ✅ Match |
| **Cooperative routing** | "50 cooperative agents" with "shared knowledge" | Shared replay buffer + per-agent replay | ✅ Match |
| **Partial observability** | "Each node has an agent... learns to optimize under **partial observability**" | All agents receive **full global graph** embedding from GAT forward pass | ⚠️ Gap — full observability used |
| **Independent policy networks** | Implied ("each node has its own policy") | Each `NodeAgent` has its own `GAT_DTAware` policy and target networks | ✅ Match |
| **MA epsilon decay** | Not specified | `MA_EPS_DECAY = 1500` (episode-based, not step-based) | ✅ Correct design |
| **MA target update** | Not specified | `MA_TARGET_UPDATE = 10` episodes with soft update (τ=0.005) | ✅ Match |

---

### 2.6 Experimental Results Reported in Paper

| Metric | Paper Reported (MA DT-GCN) | Paper Reported (GCN Baseline) | Difference | Tracked in Code |
|---|---|---|---|---|
| Average Reward | +1.017 | −8.471 | +9.488 | `eps_reward` in metrics |
| Success Rate | 94% | 76% | +18 pp | `success_rate` in metrics |
| Avg Path Length | 3.54 hops | 27.44 hops | −23.9 hops | `path_length` in metrics |
| Avg Jammed Steps | 0.31 | 1.40 | −1.09 steps | `jammed_steps` in metrics |
| Training Time | 34.0 min | 122.1 min | −72.1% faster | `time.time()` in training scripts |

---

## 3. Gaps and Discrepancies Found

### ⚠️ Gap 1: "Partial Observability" Not Fully Enforced
- Paper states: *"Each node has an agent embedded with it. The agent learns to optimize local and global performance under partial observability"*
- In the implementation, all agents receive the **full global graph** through `GAT_DTAware` forward pass
- True partial observability would restrict each agent's observation to its k-hop neighborhood
- **Risk:** Significant architectural gap from the paper's claim

### ⚠️ Gap 2: PDR (Packet Delivery Ratio) Metric Implementation
- Paper discusses *"average success rate (packet delivery ratio)"* as a key metric
- In the code, `step_pdr` is set to bandwidth value: a **proxy**, not actual PDR
- Actual PDR would require counting successful vs. attempted packet transmissions
- **Risk:** The PDR metric in the paper and the code measure different things

### ⚠️ Gap 3: RSS Connectivity Threshold Not Used in Edge Pruning
- `config.py` defines `RSS_CONNECTIVITY_THRESHOLD = -85` (min RSS for edge to exist)
- In `mesh_twin.py`, edges are **never removed** based on RSS — only weights change
- Paper implies dynamic link availability based on signal quality
- **Risk:** Network topology in simulation is static; paper implies dynamic link removal

### ⚠️ Gap 4: `edge_attr` Not Passed to GATConv Forward Pass
- Paper says GAT uses edge weights for "message passing" between neighbor nodes
- In `gat_dt_aware.py`, `GATConv(node_feat_dim, conv1_out, edge_dim=1)` declares `edge_dim=1`
- But in `forward()`, `self.conv1(batch.x, batch.edge_index)` — **no `edge_attr` is passed**
- DT-derived edge weights (latency/capacity from radio maps) are not used in attention computation
- **Risk:** This is the most critical bug — edge weights from DT are effectively ignored during attention

### ⚠️ Gap 5: Default Episode Count Mismatch
- Paper clearly states experiments used 3,000 episodes
- Training scripts default to 500 episodes (`--episodes 500`)
- The 3,000 count is achievable via CLI arg `--episodes 3000` but may confuse reproduction
- **Risk:** Low — configurable, just needs documentation

### ⚠️ Gap 6: Paper Terminology Inconsistency (GCN vs GAT)
- Paper title uses "GCN" but methods section describes "two Graph Attention Network layers"
- Code correctly uses `GATConv` and classes are now named `GAT_DTAware`
- **Risk:** Readers comparing title ("GCN") to code ("GAT") may be confused, but the methods section is accurate

---

## 4. Suggested Improvements

### 🔧 Fix 1: Pass `edge_attr` to GATConv (Critical)
```python
# In gat_dt_aware.py forward():
out = self.conv1(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
out = F.selu(out)
out = self.conv2(out, batch.edge_index, edge_attr=batch.edge_attr)
```
This ensures DT-derived edge weights actually influence the attention mechanism — a core paper claim.

### 🔧 Fix 2: Implement Dynamic Edge Pruning Based on RSS
```python
# In mesh_twin.py update_link_weights():
if avg_rss < RSS_CONNECTIVITY_THRESHOLD:
    self.graph.remove_edge(u, v)  # Link drops below usability
```
Implements the paper's claim that low-RSS links become inactive.

### 🔧 Fix 3: Add Proper Partial Observability for Multi-Agent
```python
# In multi_agent.py, restrict each agent's view:
def get_local_observation(self, agent_node, k_hops=2):
    subgraph = nx.ego_graph(self.env.graph, agent_node, radius=k_hops)
    # Build PyG Data from subgraph only
    ...
```
Aligns implementation with the paper's claim of partial observability.

### 🔧 Fix 4: Implement Actual PDR Metric
```python
# Track actual successful vs attempted transmissions per episode
def compute_pdr(successful_deliveries, total_attempts):
    return successful_deliveries / total_attempts if total_attempts > 0 else 0.0
```
Replace the bandwidth proxy with packet-level success tracking.

### 🔧 Fix 5: Update Default Episode Count
```python
# In train_integrated.py and train_multi_agent.py:
parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
```
Match the paper's stated experimental setup of 3,000 episodes.

---

## 5. Summary Alignment Table

| Paper Section | Implementation File | Alignment |
|---|---|---|
| Digital Twin (Radio Map) | `digital_twin/mesh_twin.py` | ✅ Fully aligned |
| Digital Twin (Anomaly Detection) | `digital_twin/anomaly_bridge.py` | ✅ Fully aligned |
| Digital Twin (Jammer Detection) | `digital_twin/anomaly_bridge.py` | ✅ Fully aligned |
| 7-Dim Node Features | `environment/hybrid_env.py` | ✅ Fully aligned |
| GAT Architecture | `models/gat_dt_aware.py` | ✅ Correctly named and implemented |
| DQN with Experience Replay | `models/gat_dt_aware.py` | ✅ Fully aligned |
| Epsilon-Greedy Exploration | `models/gat_dt_aware.py` + `models/multi_agent.py` | ✅ Fully aligned |
| Enhanced Reward Function | `environment/enhanced_reward.py` | ✅ Fully aligned |
| Multi-Agent (50 agents) | `models/multi_agent.py` | ✅ Aligned |
| Shared Experience Replay | `models/multi_agent.py` | ✅ Aligned |
| Training Time Benchmarking | `train_integrated.py`, `train_multi_agent.py` | ✅ Implemented |
| Reproducibility (Seeds) | All training scripts | ✅ `--seed` CLI arg implemented |
| Partial Observability | `models/multi_agent.py` | ⚠️ Not enforced — global view used |
| Edge RSS-based pruning | `digital_twin/mesh_twin.py` | ⚠️ Threshold defined but not applied |
| Edge weights in GAT attention | `models/gat_dt_aware.py` | ❌ Bug: `edge_attr` not passed to GATConv |
| PDR as evaluation metric | `models/gat_dt_aware.py` | ⚠️ Bandwidth used as proxy, not actual PDR |
| 94% vs 76% success rate | Metrics tracking | ⚠️ Reproducibility depends on seed + 3000 episodes |

---

## 6. Priority Fix Order

| Priority | Fix | Impact | Effort |
|---|---|---|---|
| 🔴 1 | Pass `edge_attr` to GATConv | Critical — DT edge weights currently ignored | ~15 min |
| 🟡 2 | Dynamic edge pruning (RSS threshold) | Medium — aligns with paper's topology claims | ~20 min |
| 🟡 3 | Partial observability | Medium — conceptual gap with paper | ~30 min |
| 🟢 4 | Actual PDR metric | Low — current proxy works for ranking | ~25 min |
| 🟢 5 | Default 3000 episodes | Low — trivial one-line change | ~1 min |

---

## 7. Conclusion

The project implementation is **strongly aligned** with papers on all major architectural and algorithmic components. The Digital Twin pipeline, GAT-based model, 7-dimensional node features, enhanced reward function, shared experience multi-agent DQN framework, training time benchmarking, and reproducibility seeds are all implemented correctly.

The **most critical bug** is Fix 1: `edge_attr` not being passed to the GAT convolution layers — this means the DT-derived edge weights (latency/capacity from radio maps) are not actually influencing the attention mechanism, which is a core claim of the paper.

The **most significant conceptual gap** is partial observability — the paper says agents operate under partial observability, but all agents receive full global graph information.

Addressing Fix 1 first will bring the implementation into compliance with the paper's core technical contribution.
