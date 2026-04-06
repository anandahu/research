# Resilient Routing in Wireless Mesh Networks using Digital Twin and GNN-Based Reinforcement Learning

**Anandakrishnan K V**

Department of Computer Science and IT, School of Computing, Amrita Vishwa Vidyapeetham, Kochi, Kerala, India

Guide: **Drupad M D**

---

## Abstract

A Wireless Mesh Network (WMN) is a communications network made up of multiple radio nodes organized in a mesh topology used for connecting dynamically with each other. Their flexibility in dynamic connectivity and multi-hop capabilities can attract major threats. Particularly, jamming attacks can severely degrade network performance by disrupting routing and data transmission. Traditional routing protocols lack the adaptability to operate effectively in such dynamic and hostile environments.

This dissertation proposes an integrated resilience framework for WMNs that combines Digital Twin (DT) modelling, Graph Attention Networks (GAT), and Multi-Agent Deep Q-Networks (DQN). A Digital Twin simulates realistic network behaviour and jamming scenarios by maintaining a virtual copy of the physical network that is continuously updated with real-time radio environment data. The Digital Twin ingests environmental radio maps containing Received Signal Strength (RSS) measurements, which are linked to link-level performance indicators including latency and bandwidth capacity. Anomaly detection and jamming analysis modules augment the network graph with additional information, constructing a seven-dimensional feature vector for each node.

A Graph Attention Network model captures the topological and traffic features by embedding the entire network topology through attention-based message passing between neighbouring nodes. The attention mechanism enables nodes to selectively focus on important neighbours based on their relevance to routing, facilitating the propagation of information about node failures and jamming attacks across the network. The output embeddings are used to estimate Q-values for routing decisions.

A multi-agent deep Q-network utilizes these embeddings to learn adaptive routing strategies under varying jamming conditions. Each of the 50 nodes in the network has its own agent with independent policy and target networks, while sharing experiences through a shared replay memory pool. The framework uses an epsilon-greedy exploration strategy with an enhanced reward function that incorporates anomaly dampening, jamming penalties, and resilience bonuses.

Experimental results demonstrate that the proposed Multi-Agent Digital Twin Graph Attention Network (MA DT-GCN) model significantly outperforms the single-agent GAT baseline across 3,000 training episodes. The MA DT-GCN achieved an average reward of 0.93 compared to the baseline's −17.78, reduced average path length from 50.85 hops to 3.67 hops (a 92.8% reduction), decreased average jammed steps from 1.25 to 0.29 (a 76.8% reduction), and converged 52.8% faster while maintaining a 100% success rate. These results validate the effectiveness of integrating Digital Twin information with multi-agent reinforcement learning for resilient routing in adversarial wireless mesh network environments.

**Keywords:** Digital Twin, Wireless Mesh Networks, Graph Attention Network, Deep Q-Network, Reinforcement Learning, Jamming Detection, Anomaly Detection, Resilient Routing, Multi-Agent Systems.

---

## 1. Introduction

### 1.1 Background

Wireless communication networks have greatly influenced the advancement of interconnectivity of technology without a physical medium. The proliferation of wireless devices in the modern era has created an unprecedented demand for reliable, scalable, and resilient networking solutions. From industrial automation to disaster response, the need for networks that can self-organize, self-heal, and adapt to changing conditions has never been greater. However, the open nature of wireless communication channels introduces significant security challenges, particularly in the form of jamming attacks that can disrupt normal network operations.

The convergence of artificial intelligence and networking has opened new avenues for addressing these challenges. Machine learning techniques, particularly deep reinforcement learning, have demonstrated remarkable capabilities in dynamic decision-making scenarios. When combined with graph-based learning methods that can naturally represent network topologies, these techniques offer a promising path toward truly intelligent and adaptive network management.

This dissertation explores the integration of three cutting-edge technologies — Digital Twins, Graph Attention Networks, and Multi-Agent Deep Q-Networks — to create a comprehensive resilience framework for Wireless Mesh Networks. The proposed framework not only detects jamming attacks in real-time but also actively adapts routing strategies to maintain network performance under adversarial conditions.

### 1.2 Wireless Mesh Networks

Wireless Mesh Networks (WMNs) represent a class of wireless networks where nodes are interconnected in a mesh topology, enabling multi-hop communication. Unlike traditional point-to-point or star topology networks, WMNs offer several distinctive advantages that make them suitable for a wide range of applications.

In a WMN, each node can communicate with multiple other nodes, creating redundant paths for data transmission. This multi-hop routing capability extends the coverage area far beyond what a single wireless device can achieve. The decentralized architecture of WMNs eliminates single points of failure, as the network can automatically reroute traffic around failed or compromised nodes. This self-healing capability makes WMNs particularly valuable in scenarios where network reliability is critical.

WMNs find applications in numerous domains including:

- **Disaster Response:** When existing infrastructure is destroyed, WMNs can be rapidly deployed to establish communication networks for emergency responders.
- **Industrial Automation:** WMNs provide reliable wireless connectivity for industrial IoT devices in manufacturing environments where wired connections may be impractical.
- **Military Communications:** The decentralized and resilient nature of WMNs makes them suitable for tactical communications in hostile environments.
- **Community Networking:** WMNs enable cost-effective internet access in rural and underserved areas by extending connectivity through mesh nodes.
- **Smart Cities:** WMNs support various smart city applications including traffic monitoring, environmental sensing, and public safety systems.

Despite these advantages, the very characteristics that make WMNs flexible also introduce vulnerabilities. The shared wireless medium, multi-hop routing, and dynamic topology create attack surfaces that malicious actors can exploit to disrupt network operations.

### 1.3 Challenges in WMN Routing

Routing in Wireless Mesh Networks presents several unique challenges that traditional routing protocols struggle to address effectively:

**Jamming Attacks:** Jamming attacks represent one of the most severe threats to WMN operations. A jammer transmits high-power signals on the same frequency band used by the network, causing interference that degrades link quality and disrupts data transmission. Jamming can be constant, deceptive, random, or reactive, each requiring different detection and mitigation strategies. The impact of jamming extends beyond the immediately affected links — it can cause cascading failures as routing protocols attempt to route traffic through jammed areas, leading to congestion and increased latency across the entire network.

**Dynamic Environment:** WMNs operate in dynamic environments where channel conditions, traffic loads, and network topology can change rapidly. Node mobility, varying interference levels, and fluctuating traffic patterns create a complex, non-stationary optimization problem. Traditional static routing protocols that rely on predefined rules cannot adapt quickly enough to these changes, resulting in suboptimal routing decisions and degraded performance.

**Scalability:** As the number of nodes in a WMN increases, the routing problem becomes increasingly complex. Mathematical optimization approaches that may work well for small networks face scalability limitations when applied to larger topologies.

**Distributed Decision Making:** The decentralized nature of WMNs requires routing decisions to be made distributedly, with each node making local decisions based on limited information about the global network state.

**Quality of Service:** Many WMN applications require guaranteed Quality of Service (QoS) in terms of latency, bandwidth, and reliability. Maintaining QoS guarantees in the presence of jamming attacks and dynamic network conditions is extremely challenging.

### 1.4 Problem Statement

Traditional routing protocols for Wireless Mesh Networks, including AODV, OLSR, and HWMP, rely on reactive or proactive approaches that use fixed metrics such as hop count or link quality to determine routing paths. These protocols lack the ability to anticipate and respond to adversarial conditions, particularly jamming attacks. When a jammer is activated, these protocols can only react after packet loss has been detected, leading to significant delays and degraded performance during the adaptation period.

While reinforcement learning-based routing approaches have shown promise, existing methods suffer from several limitations. First, many RL approaches treat the network as a simple state-action space without leveraging the graph structure of the network topology. Second, most approaches do not incorporate real-time environmental awareness, relying instead on packet-level feedback that provides limited information about the underlying causes of performance degradation. Third, single-agent RL approaches cannot effectively capture the distributed nature of routing decisions in WMNs.

The core problem addressed in this dissertation is: **How can we design a routing framework for Wireless Mesh Networks that can proactively detect and adapt to jamming attacks in real-time, while leveraging the graph structure of the network and enabling distributed decision-making?**

### 1.5 Objectives

The primary objectives of this dissertation are as follows:

1. **Design and implement a Digital Twin module** that maintains a real-time virtual replica of the physical network, enabling real-time anomaly detection and jamming analysis through comparison of expected and actual radio signals.
2. **Develop a Graph Attention Network model** that leverages the graph structure of the network topology to generate node embeddings enriched with Digital Twin-derived features, enabling topology-aware routing decisions.
3. **Implement a Multi-Agent Deep Q-Network framework** where each node in the network has its own learning agent, enabling distributed and cooperative routing decisions that respect the decentralized nature of WMNs.
4. **Design an enhanced reward function** that incorporates anomaly scores, jamming penalties, and resilience bonuses to guide agents toward routes that are not only short but also resilient to jamming attacks.
5. **Evaluate the proposed framework** through extensive simulation experiments comparing the multi-agent Digital Twin-enhanced model against a baseline GAT model without Digital Twin awareness.

### 1.6 Scope of the Work

This dissertation focuses on the design, implementation, and evaluation of a resilient routing framework for WMNs. The scope encompasses:

- A 50-node wireless mesh network topology generated using the BRITE (Boston University Representative Internet Topology Generator) topology generator
- Digital Twin-based anomaly detection and jamming analysis using datasets provided by the Digital Twin Radio Environment framework
- Graph Attention Network-based state representation using PyTorch Geometric's GATConv implementation
- Multi-Agent Deep Q-Network with 50 independent agents, one per network node
- Simulation-based evaluation over 3,000 training episodes

The work is limited to simulation-based evaluation and does not include deployment on physical hardware testbeds. The jamming model uses pre-generated radio maps rather than real-time radio measurements.

---

## 2. Related Works

### 2.1 Reinforcement Learning in Network Routing

Routing in Wireless Mesh Networks is typically modeled as a Markov Decision Process (MDP), which provides a mathematical framework for decision-making in stochastic environments. In this formulation, the network state includes information about topology, link quality, and traffic conditions; actions correspond to routing decisions (selecting the next hop for a packet); and rewards reflect the quality of the routing decision (e.g., successful delivery, low latency, avoiding jammed areas).

Reinforcement Learning (RL) techniques have shown considerable promise in adaptive, real-time decision-making in dynamic environments. Unlike traditional optimization methods that require complete knowledge of the environment, RL agents learn optimal policies through interaction with the environment, making them well-suited for dynamic network conditions where the environment model is unknown or constantly changing.

#### 2.1.1 Q-Learning and Deep Q-Networks

Q-learning is a model-free RL algorithm that learns the value of taking a specific action in a given state. The Q-value function Q(s, a) represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter. The Q-value update rule is:

> Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') − Q(s, a)]

where α is the learning rate, γ is the discount factor, r is the immediate reward, and s' is the next state.

Deep Q-Networks (DQN), introduced by Mnih et al. in 2015, extend Q-learning by using deep neural networks to approximate the Q-value function, enabling the algorithm to handle high-dimensional state spaces. DQN introduced two key innovations: experience replay (storing transitions in a buffer and sampling randomly for training) and a target network (a separate network used for computing target Q-values, updated periodically) that significantly improve training stability.

#### 2.1.2 Deep Reinforcement Learning for Network Routing

Pappone et al. (2023) proposed a framework that integrates Graph Neural Networks with Multi-Agent Deep Reinforcement Learning (DRL) to overcome a core limitation of traditional RL: the inability to generalize to unfamiliar network topologies without significant retraining. By encoding the network state using GNNs, their model captures the graph's permutation-invariant structure, enabling agents to minimize flow collisions and adapt to real-time topology changes dynamically. Their work demonstrated that GNN-based state representations enable RL agents to transfer learned policies across different network topologies, significantly reducing the need for retraining.

Similarly, Jin et al. (2025) researched a Deep Reinforcement Learning-based Resilient Routing (DRLRR) approach for urban emergency communication networks. To counter the overestimation bias of conventional Q-learning, they used Double Deep Q-Networks (DDQN), enabling stable route optimization despite excessive node degradation scenarios. Their approach integrates node and link state information into a specialized reward function, ensuring stable data transmission during disasters.

### 2.2 Graph Neural Networks for Networking

Graph Neural Networks (GNNs) have emerged as a powerful tool for learning on graph-structured data, making them naturally suited for networking applications where the network topology can be represented as a graph. GNNs operate by iteratively aggregating information from neighboring nodes, allowing each node to build a representation that captures both its local features and its structural context within the graph.

#### 2.2.1 Graph Convolutional Networks

Graph Convolutional Networks (GCNs), introduced by Kipf and Welling in 2017, gained popularity as an efficient spectral approximation of convolution on graphs. Each node's representation is iteratively refined by aggregating the features of its local neighbours, normalized by the degrees of related nodes. The GCN layer-wise propagation rule is:

> H^(l+1) = σ(D̃^(−½) Ã D̃^(−½) H^(l) W^(l))

where Ã = A + I_N is the adjacency matrix with added self-connections, D̃ is the degree matrix of Ã, W^(l) is the layer-specific trainable weight matrix, and σ is the activation function. During convolution, the GCN assumes a constant graph structure and uses only the adjacency of the graph to determine aggregation weights instead of learned pairwise significance scores. While computationally efficient, this uniform weighting scheme limits the model's ability to distinguish between more and less important neighbors.

#### 2.2.2 Graph Attention Networks

Graph Attention Networks (GATs), introduced by Veličković et al. in 2018, address the limitations of GCNs by introducing a self-attention mechanism that computes attention coefficients α_ij indicating how important neighbour j is to node i:

> α_ij = exp(LeakyReLU(a^T[Wh_i ∥ Wh_j])) / Σ_k∈N_i exp(LeakyReLU(a^T[Wh_i ∥ Wh_k]))

Unlike GCN, which assigns uniform weights during aggregation, GAT selectively weights each neighbour's contribution through a learnable attention mechanism. This makes it particularly suited for environments where link reliability varies due to jamming or interference. Selective aggregation allows GAT to better capture the importance of high-quality, non-jammed links during routing, which is a core motivation for its adoption in this framework.

GATs also support multi-head attention, where K independent attention mechanisms are applied and their outputs are concatenated or averaged. This stabilizes the learning process and allows the model to attend to information from different representation subspaces at different positions.

### 2.3 Digital Twin for Radio Environments

The concept of a Digital Twin — a real-time virtual replica of a physical system — has gained significant traction in the context of wireless network management. A Digital Twin for radio environments typically consists of:

- **Radio Map:** A spatial representation of the radio environment containing Received Signal Strength (RSS) values at grid points across the coverage area.
- **Path Loss Model:** A mathematical model that predicts signal attenuation based on distance and environmental factors.
- **Anomaly Detection:** Modules that compare the expected radio environment with actual measurements to identify deviations indicating jammers or other anomalies.

Krause et al. (2023) proposed a novel approach to anomaly detection in radio environments using Digital Twins. Rather than anchoring detection to patterns mined from past data, their system maintains a physics-based virtual replica of the network running in real time. A ray-tracing engine inside the replica computes expected RSSI readings across each link under normal conditions. Any meaningful gap between predicted and measured signals is flagged — no attack examples are needed for training.

Crucially, their framework can distinguish jamming from ordinary fading, which is exactly where conventional statistical detection tends to fail. The dataset generated by Krause et al. forms the basis for the Digital Twin component in the proposed framework. It provides 30,000 radio map scenarios per dataset, including both normal and jammed conditions.

### 2.4 Multi-Agent Systems in Network Optimization

Multi-Agent Reinforcement Learning (MARL) extends traditional RL by enabling multiple agents to learn and make decisions simultaneously within a shared environment. In the context of network routing, MARL is a natural fit because routing decisions are inherently decentralized.

Cooperative MARL introduces mechanisms for agents to share information and coordinate their learning. Common approaches include shared experience replay, parameter sharing, and communication protocols.

The work by Bhavanasi, Pappone, and Esposito (2023) on "Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning" provides the foundational architecture for the multi-agent component of this framework. Their RP15 model implements one DQN agent per node in a 50-node Barabási-Albert network.

### 2.5 Gap Analysis

The review of existing literature reveals several significant gaps that motivate the proposed framework:

**Gap 1: Limited Integration of DT and Routing.** Pappone et al. (2023) built a solid routing foundation using GNNs and multi-agent DRL, yet the work overlooks the subtleties of physical-layer threats by anchoring itself to standard network metrics. The agents have no awareness of the underlying physical layer conditions.

**Gap 2: Detection Without Intervention.** Krause et al. (2023) leverage digital twins for detection but deliberately stop at observation, leaving the door closed on any form of active network intervention.

**Gap 3: Missing Feature Enrichment.** Existing work has yet to meaningfully explore the tight integration of Digital Twin anomaly detection and GNN-based routing — particularly feeding high-fidelity anomaly scores from a Digital Twin directly into a Graph Attention Network as live, dynamic features.

**Gap 4: Lack of Attention-Based Link Assessment.** Most GNN-based routing approaches use GCN layers that assign uniform weights to all neighbours during aggregation. Attention mechanisms can significantly improve routing decisions in varying link quality environments.

This dissertation addresses these gaps by integrating all three capabilities: DT-based anomaly detection, GAT-based topology embedding with DT-enriched features, and multi-agent DRL for distributed routing.

| Reference | Approach | Strengths | Limitations |
|---|---|---|---|
| Pappone et al. (2023) | GNN + Multi-Agent DRL | Topology-aware routing, Generalization | No physical-layer threat awareness, No DT integration |
| Jin et al. (2025) | DDQN for Emergency Routing | Resilient routing under node degradation | Single-agent, No graph structure utilization |
| Kipf & Welling (2017) | GCN | Efficient graph learning | Uniform neighbor weighting |
| Veličković et al. (2018) | GAT | Selective neighbor weighting via attention | Not designed for routing |
| Krause et al. (2023) | DT Anomaly Detection | Physics-based jamming detection | Detection only — no active routing |
| **This Work** | **DT + GAT + Multi-Agent DQN** | **Integrates detection, representation, and routing** | **Simulation-based evaluation only** |

---

## 3. Dataset Description

### 3.1 Network Topology Dataset

The network topology is generated using the BRITE (Boston University Representative Internet Topology Generator) tool. The generated topology consists of **50 nodes** and **100 bidirectional edges**, following the Barabási-Albert preferential attachment model. The topology file (50nodes.brite) provides node positions in a 2D coordinate space (ranging 0–99) and edge connections with default weight and capacity attributes.

Node positions are scaled from BRITE coordinates (0–99) to Digital Twin coordinates (0–40) using a scaling factor of 0.4. This mapping aligns the network nodes with the 41×41 radio map grid used by the Digital Twin.

### 3.2 Digital Twin Radio Environment Dataset

The Digital Twin Radio Environment dataset, developed by Krause et al. (2023), provides realistic radio propagation data for anomaly detection and jamming analysis. The dataset is stored as Python pickle files and comprises three components:

| Dataset Component | File Format | Contents | Size per Dataset |
|---|---|---|---|
| Radio Map Dataset | fspl_RMdataset{N}.pkl | 41×41 RSS grid arrays (dBm) | 30,000 scenarios |
| Measurement Dataset | fspl_measurements{N}.pkl | Measurement differences, jammer labels, coordinates | 30,000 scenarios |
| Path Loss Dataset | fspl_PLdataset{N}.pkl | Path loss maps | 30,000 scenarios |

**Radio Maps:** Each scenario in the radio map dataset contains a 41×41 grid array of Received Signal Strength (RSS) values in dBm. The RSS values represent the signal strength at each grid point, accounting for path loss, shadowing, and (in jammed scenarios) interference from active jammers. The grid resolution corresponds to a physical measurement spacing of approximately 10 meters.

**Measurement Data:** Each scenario includes 25 measurement points, each with:
- Measurement coordinates (x, y positions in the DT grid)
- Measurement differences (the discrepancy between expected and actual RSS values)
- Jammer labels (boolean indicating whether the scenario includes active jammers)
- Jammer positions (x, y coordinates of active jammers when present)

**Scenario Distribution:** Approximately 50% of the 30,000 scenarios include one or more active jammers, while the remaining 50% represent normal (unjammed) conditions. The jammed scenarios use the Free-Space Path Loss (FSPL) model augmented with jammer interference patterns.

### 3.3 Training Data Split

For anomaly detection training, the Anomaly Bridge module is trained on measurement differences from **2,000 normal (non-jammed)** samples. This unsupervised approach establishes a statistical baseline of normal behaviour, against which anomalous (jammed) conditions can be identified using a 2-sigma threshold — requiring no labeled attack examples for training.

During routing agent training, the environment randomly selects scenarios from the full dataset with **50% probability** of selecting a jammed scenario and 50% probability of selecting a normal scenario, ensuring balanced exposure to both conditions over 3,000 training episodes.

---

## 4. Methodology

### 4.1 System Overview

The proposed Multi-Agent Digital Twin Graph Attention Network (MA DT-GCN) framework consists of three interconnected modules:

1. **Digital Twin Module** — Environmental awareness through radio map processing, anomaly detection, and jammer identification
2. **Graph Attention Network (GAT)** — Topology-aware feature extraction using attention-based message passing
3. **Multi-Agent Deep Q-Network (DQN)** — Distributed routing through 50 independent cooperative agents

### 4.2 Digital Twin Module

The Digital Twin module is the environmental awareness layer of the framework. It is implemented through three primary components.

#### 4.2.1 Mesh Digital Twin

The Mesh Digital Twin maintains a virtual copy of the physical wireless mesh network. It ingests environmental radio maps (41×41 grid arrays of RSS values in dBm) and updates link-level quality metrics for all edges in the network.

For each edge (u, v), the RSS values at both endpoint positions are averaged, and two transformations are applied:

**RSS to Latency (Sigmoidal Transformation):**

> latency = 1 / (1 + exp((RSS − RSS_ref) / 30)),  latency ∈ [0.01, 0.99]

where RSS_ref = −70 dBm is the reference signal level. This sigmoidal mapping captures the non-linear relationship between signal strength and link performance.

**RSS to Capacity (Linear Normalization):**

> capacity = (RSS + 100) / 50,  capacity ∈ [0.01, 1.0]

The updated latency and capacity values serve as edge weights in the network graph, directly influencing shortest path computations and providing edge attributes for the GAT attention mechanism.

#### 4.2.2 Anomaly Bridge

The Anomaly Bridge converts Digital Twin measurement differences into per-node anomaly scores.

**Training Phase:** The Anomaly Bridge is trained on measurements from non-jammed scenarios. It computes the mean (μ) and standard deviation (σ) of absolute measurement differences across all normal samples and establishes a 2-sigma threshold:

> threshold = μ + 2σ

**Scoring Phase:** For each scenario, the absolute measurement differences are compared against the threshold. Scores are computed on a 0-to-1 scale, where 0 indicates normal behaviour and values approaching 1 indicate increasingly anomalous conditions.

**Spatial Mapping:** Measurement point scores are mapped to mesh node scores using nearest-neighbour interpolation with distance-based exponential decay.

#### 4.2.3 Jammer Detector

The Jammer Detector provides per-node jamming probability estimates based on the distance to known jammer positions, operating with a configurable jamming radius (default: 10 units in DT coordinates).

- **Hard Detection:** Nodes within the jamming radius are classified as definitively jammed (probability = 1.0).
- **Soft Detection:** Nodes beyond the jamming radius receive exponentially decaying jamming probabilities based on distance from the nearest jammer.

#### 4.2.4 Seven-Dimensional Node Feature Vector

The Digital Twin module produces a comprehensive seven-dimensional feature vector for each node:

| Index | Feature | Source | Description |
|---|---|---|---|
| 0 | is_source | Environment | 1.0 if node is current packet position |
| 1 | is_dest | Environment | 1.0 if node is the routing target |
| 2 | avg_latency | Digital Twin | Mean latency to all neighbour nodes |
| 3 | avg_bandwidth | Digital Twin | Mean capacity to all neighbour nodes |
| 4 | anomaly_score | Anomaly Bridge | Anomaly score from DT measurements (0–1) |
| 5 | jam_probability | Jammer Detector | Estimated jamming probability (0–1) |
| 6 | neighbor_jam_avg | Jammer Detector | Mean jamming probability across neighbours |

The baseline GAT model uses only the first 4 features (with features 4–6 zeroed out), enabling a clean evaluation of the DT contribution.

### 4.3 Graph Attention Network Architecture

The GAT_DTAware model consists of four layers:

1. **First GATConv Layer:** Input (7-dim features) → 16-dim embeddings, with edge_dim=1 for DT-derived edge weights.
2. **Second GATConv Layer:** 16-dim → 25-dim, with SELU activation between layers.
3. **First FC Layer:** Flattened embeddings (50 × 25 = 1,250) → 50-dim.
4. **Second FC Layer:** 50-dim → Q-values (num_neighbors for multi-agent, or 50 for single-agent).

| Layer | Type | Input Dim | Output Dim | Parameters |
|---|---|---|---|---|
| conv1 | GATConv | 7 | 16 | edge_dim=1 |
| conv2 | GATConv | 16 | 25 | edge_dim=1 |
| fc1 | Linear | 1,250 | 50 | — |
| fc2 | Linear | 50 | num_neighbors | — |

**Edge Attribute Integration:** DT-computed latency values are passed as edge_attr to the GATConv layers, allowing the attention mechanism to consider link quality when computing attention coefficients. Nodes with high-quality links receive higher attention weights, while degraded links receive lower weights.

**Action Masking:** Q-value outputs are masked so that only valid actions (neighbours of the current node) are selectable. Invalid actions have Q-values set to negative infinity.

### 4.4 Multi-Agent Deep Q-Network

The Multi-Agent system implements cooperative routing with 50 independent DQN agents, one per network node.

#### 4.4.1 Node Agent Architecture

Each NodeAgent consists of:
- **Policy Network:** GAT_DTAware model for action selection
- **Target Network:** Separate GAT_DTAware model for computing target Q-values
- **Replay Memory:** Per-agent buffer with capacity 20,000 transitions
- **Optimizer:** Adam optimizer with learning rate 0.001

#### 4.4.2 Shared Experience Replay

A global SharedReplayMemory pool (capacity 50,000 transitions) enables cooperative learning. The pool implements compatibility filtering — agents can only sample transitions from other agents with the same number of neighbours.

Sampling strategy: 60% own transitions + 40% shared transitions per training batch.

#### 4.4.3 Target Network Updates

Target networks use soft updates (Polyak averaging) every 10 episodes:

> θ_target ← τ · θ_policy + (1 − τ) · θ_target,  τ = 0.005

| Parameter | Single-Agent | Multi-Agent |
|---|---|---|
| Learning Rate | 0.001 | 0.001 |
| Discount Factor (γ) | 0.99 | 0.99 |
| Batch Size | 128 | 32 |
| Replay Memory Size | 5,000 | 20,000/agent + 50,000 shared |
| Epsilon Start → End | 0.95 → 0.001 | 0.95 → 0.001 |
| Epsilon Decay | 1,000 steps | 1,500 episodes |
| Target Update | Every 14,000 steps (hard) | Every 10 episodes (soft, τ=0.005) |
| Loss Function | Smooth L1 (Huber) | Smooth L1 (Huber) |
| Gradient Clipping | Max norm = 1.0 | Max norm = 1.0 |

### 4.5 Reward Function Design

The reward function consists of four components balancing routing efficiency with resilience.

#### 4.5.1 Base Reward

Based on A* shortest path progress:
- **Reached target optimally:** +1.01
- **Reached target suboptimally:** −1.51
- **Progress toward target:** d_old − d_new (positive if closer)
- **Moved away from target:** −1.0
- **Timeout (path > 10 × num_nodes):** −1.0

#### 4.5.2 Anomaly Dampening

> R_dampened = R_base × (1 − α × anomaly_score),  α = 0.3

#### 4.5.3 Jamming Penalty

> R_jam = −λ  (if current node is jammed),  λ = 0.5

#### 4.5.4 Resilience Bonus

> R_resilience = +0.3  (if reached target successfully, jammed nodes exist in scenario, and path avoids all of them)

| Parameter | Symbol | Value | Effect |
|---|---|---|---|
| Anomaly Dampening | α | 0.3 | Reduces reward in anomalous areas |
| Jamming Penalty | λ | 0.5 | Fixed penalty for jammed nodes |
| Resilience Bonus | — | 0.3 | Bonus for clean (jam-free) paths |

### 4.6 Training Algorithm

**Algorithm: Resilient Routing with MA DT-GCN**

1. **Initialize:** DT radio map, MA-DQN agent replay buffers B_i, and GAT networks with arbitrary weights.
2. **For each training episode:**
   a. **Reset:** Network topology and initialize packet at source node.
   b. **Digital Twin Update:** Retrieve real-time RSS map and compute anomaly/jamming probabilities.
   c. **State Representation:** For each node i, construct the 7-dimensional feature vector x_i.
   d. **While** packet not at destination and max steps not reached:
      - i. **GAT Attention:** Compute node embeddings h_i by aggregating neighbour information using learned attention coefficients.
      - ii. **Action Selection:** With probability ε select random neighbour; otherwise select neighbour with highest Q-value.
      - iii. **Execute** action (forward packet to selected neighbour).
      - iv. **Observe** reward and next state embeddings.
      - v. **Store** transition in replay buffer B_i and shared pool.
      - vi. **Experience Replay:** Sample mini-batch from B_i ∪ shared pool.
      - vii. **Policy Update:** Gradient descent on Huber loss to update weights θ_i.
   e. **Decay** exploration rate ε.
   f. **Soft update** target networks every 10 episodes.

### 4.7 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python 3.10+ | Core implementation |
| Deep Learning | PyTorch | Neural network training |
| Graph Neural Networks | PyTorch Geometric (PyG) | GATConv layers, graph data |
| Graph Processing | NetworkX | Topology, shortest paths |
| RL Environment | OpenAI Gym | Environment interface |
| Scientific Computing | NumPy | Array operations, radio maps |
| Data Analysis | Pandas | Metrics recording, CSV output |
| Visualization | Matplotlib | Training graphs |
| Topology Generation | BRITE | Network topology files |
| GPU Acceleration | CUDA | Optional GPU training |

---

## 5. Results and Discussion

### 5.1 Experimental Setup

The experiments are conducted on a 50-node WMN (100 bidirectional edges) generated using BRITE. Two models are compared:

1. **MA GAT+DT (Proposed):** Multi-Agent GAT with full 7-dimensional DT-enriched features and 50 independent agents.
2. **GAT Baseline:** Single-Agent GAT using only 4 features (DT features zeroed out).

Both models are trained for 3,000 episodes with random seed 42 for reproducibility. Maximum steps per episode: 500.

### 5.2 Performance Summary

| Metric | MA GAT+DT | GAT Baseline | Improvement |
|---|---|---|---|
| Average Reward | 0.93 | −17.78 | +18.71 |
| Success Rate | 100% | 93% | +7 pp |
| Average Path Length | 3.67 hops | 50.85 hops | −92.8% |
| Average Jammed Steps | 0.29 | 1.25 | −76.8% |
| Training Time | 35.51 min | 75.21 min | 52.8% faster |
| Average PDR | 100% | 100% | — |

### 5.3 Reward and Routing Stability

The MA GAT+DT model demonstrates significantly faster convergence and higher final reward values compared to the GAT Baseline. The multi-agent model begins generating positive rewards within the first few hundred episodes, indicating that the combination of distributed decision-making and DT-enriched features enables rapid learning of effective routing strategies.

In contrast, the GAT Baseline model exhibits much slower convergence and consistently lower rewards throughout training. Without Digital Twin awareness, the baseline agent frequently routes through jammed areas, incurring penalties, and takes substantially longer paths.

The stability of the MA GAT+DT reward curve (low variance in the moving average) indicates that multi-agent coordination and shared experience replay effectively prevent catastrophic forgetting and policy oscillation.

### 5.4 Path Efficiency Analysis

The MA GAT+DT model converges to an average path length of **3.67 hops**, compared to the baseline's **50.85 hops** — a reduction of over 92%. This demonstrates the effectiveness of DT features in providing the agent with environmental awareness for efficient routing.

The baseline's high average path length indicates that without DT features, the agent struggles to find direct routes, often wandering through the network. The multi-agent architecture, where each node makes an informed local decision, enables more efficient hop-by-hop routing compared to a single centralized agent.

### 5.5 Jamming Avoidance Analysis

The MA GAT+DT model averages only **0.29 jammed steps** per episode, compared to **1.25 for the baseline** — a reduction of 76.8%.

This improvement is directly attributable to the Digital Twin integration. The anomaly scores and jamming probability features (dimensions 4–6 of the node feature vector) provide the GAT attention mechanism with real-time information about which nodes are affected by jamming. The baseline, with DT features zeroed out, can only learn to avoid jammed areas through trial-and-error — discovering jammed nodes after routing through them and incurring penalties. This reactive approach is inherently less effective than the proactive avoidance enabled by DT awareness.

### 5.6 Convergence and Training Efficiency

The MA GAT+DT model completes training in **35.51 minutes**, while the GAT Baseline requires **75.21 minutes** — 52.8% faster. Contributing factors:

1. **Shorter Episodes:** Because the MA GAT+DT model finds paths more quickly (3.67 vs. 50.85 hops), each episode requires fewer environment steps.
2. **Better Exploration:** DT features provide meaningful gradient information from the start, reducing exploration time.
3. **Shared Learning:** The shared experience replay pool enables agents to learn from each other's experiences.

### 5.7 Success Rate and Reliability

The MA GAT+DT model achieves a perfect **100% success rate**, while the GAT Baseline achieves **93%**. The 7 percentage point improvement demonstrates that Digital Twin awareness and multi-agent coordination improve routing reliability — the agent never fails to deliver a packet when equipped with environmental awareness.

Both models achieve 100% PDR for successfully delivered packets. The difference lies in whether a route can be found within the step limit.

### 5.8 Detailed Improvement Analysis

| Category | Metric | MA GAT+DT | GAT Baseline | Change |
|---|---|---|---|---|
| Routing Quality | Average Reward | 0.93 | −17.78 | +18.71 |
| Routing Quality | Average Path Length | 3.67 | 50.85 | −92.8% |
| Resilience | Jammed Steps | 0.29 | 1.25 | −76.8% |
| Resilience | Success Rate | 100% | 93% | +7 pp |
| Efficiency | Training Time | 35.51 min | 75.21 min | −52.8% |
| Reliability | PDR | 100% | 100% | — |

### 5.9 Key Findings

1. **Digital Twin integration dramatically improves routing performance.** The three additional DT features (anomaly score, jam probability, neighbor jam average) provide critical environmental awareness enabling proactive jamming avoidance.
2. **Multi-agent coordination enables efficient distributed routing.** The 50 independent agents, coordinated through shared experience replay, achieve better path efficiency than the single-agent baseline while training faster.
3. **The enhanced reward function successfully balances multiple objectives.** The combination of progress-based rewards, anomaly dampening, jamming penalties, and resilience bonuses guides agents toward paths that are simultaneously short, safe, and resilient.
4. **The framework is computationally efficient.** Despite having 50 agents, the MA GAT+DT model trains faster than the single-agent baseline due to faster episode completion and better exploration.

---

## 6. Conclusion

This dissertation presented a novel framework for resilient routing in Wireless Mesh Networks that integrates Digital Twin modelling, Graph Attention Networks, and Multi-Agent Deep Q-Networks. The proposed MA DT-GCN framework addresses critical gaps in existing research by combining environmental awareness (through Digital Twin integration), topology-aware decision-making (through GAT-based state representation), and distributed routing (through multi-agent coordination).

The key contributions of this work are:

**1. Digital Twin Integration for Routing:** The framework demonstrates the effective integration of Digital Twin-derived environmental features into a GNN-based routing agent. The seven-dimensional node feature vector provides agents with comprehensive environmental awareness enabling proactive jamming avoidance rather than reactive route recovery.

**2. Attention-Based Link Quality Assessment:** The GAT architecture with edge attribute support enables the attention mechanism to consider link quality (DT-derived latency) when computing attention coefficients, allowing selective focus on high-quality non-jammed links.

**3. Multi-Agent Cooperative Routing:** The framework implements 50 independent agents with shared experience replay, enabling distributed decision-making that mirrors the decentralized nature of WMN routing.

**4. Enhanced Reward Function:** The multi-component reward function successfully balances routing efficiency with resilience objectives through anomaly dampening, jamming penalties, and resilience bonuses.

Experimental evaluation on a 50-node wireless mesh network over 3,000 training episodes demonstrates that the proposed MA GAT+DT model achieves an average reward of 0.93 (vs. baseline −17.78), reduces average path length by 92.8% (50.85 → 3.67 hops), decreases jammed steps by 76.8% (1.25 → 0.29), achieves 100% success rate (vs. 93%), and trains 52.8% faster despite having 50 agents.

These results validate the central thesis: routing the Digital Twin's environmental awareness through the attention mechanism of a GAT-based multi-agent RL system enables WMNs to build genuine resilience against jamming attacks — interference is anticipated and routed around before packet loss escalates.

---

## 7. Future Work

1. **Dynamic Topology Support:** Extending the framework to handle dynamic topologies where nodes can join, leave, or move, applicable to mobile ad-hoc networks and disaster response.

2. **Real-Time Digital Twin Updates:** Integrating real-time radio measurements from physical network deployments for true closed-loop Digital Twin operation.

3. **Advanced Jamming Models:** Incorporating reactive jammers, smart jammers that adapt to the routing protocol, and mobile jammers for more realistic evaluation.

4. **Scalability Evaluation:** Evaluating the framework on larger networks (100, 200, 500 nodes) to understand multi-agent scaling behaviour.

5. **Transfer Learning:** Investigating transferability of learned policies across different topologies and jamming scenarios, leveraging GAT's permutation invariance.

6. **Multi-Objective Optimization:** Explicitly optimizing for multiple objectives (latency, bandwidth, energy efficiency, security) using Pareto-optimal policy search.

7. **Hardware Testbed Validation:** Deploying and evaluating on a physical wireless mesh network testbed to validate simulation results.

8. **Partial Observability:** Implementing true partial observability where each agent only accesses local neighbourhood information.

---

## Appendix

### A. List of Abbreviations

| Abbreviation | Full Form |
|---|---|
| WMN | Wireless Mesh Network |
| DT | Digital Twin |
| GNN | Graph Neural Network |
| GAT | Graph Attention Network |
| GCN | Graph Convolutional Network |
| DQN | Deep Q-Network |
| DRL | Deep Reinforcement Learning |
| RL | Reinforcement Learning |
| MDP | Markov Decision Process |
| DDQN | Double Deep Q-Network |
| MA DT-GCN | Multi-Agent Digital Twin Graph Convolutional Network |
| RSS | Received Signal Strength |
| RSSI | Received Signal Strength Indicator |
| PDR | Packet Delivery Ratio |
| QoS | Quality of Service |
| IoT | Internet of Things |
| PyG | PyTorch Geometric |
| BRITE | Boston University Representative Internet Topology Generator |
| AODV | Ad hoc On-Demand Distance Vector |
| OLSR | Optimized Link State Routing |
| HWMP | Hybrid Wireless Mesh Protocol |
| FSPL | Free-Space Path Loss |

### B. Module Architecture

The codebase is organized under the `integrated_dt_gcn` package:

```
integrated_dt_gcn/
├── config.py                    # Centralized hyperparameters
├── brite_loader.py              # BRITE topology parser
├── dataset_loader.py            # DT dataset loader
├── digital_twin/
│   ├── mesh_twin.py             # Virtual network replica
│   └── anomaly_bridge.py        # Anomaly detection & jammer detection
├── environment/
│   ├── hybrid_env.py            # OpenAI Gym routing environment
│   └── enhanced_reward.py       # Multi-component reward function
└── models/
    ├── gat_dt_aware.py          # GAT model + single-agent DQN
    └── multi_agent.py           # Multi-agent coordinator
```

Training scripts:
- `train_integrated.py` — Single-agent training
- `train_multi_agent.py` — Multi-agent training
- `compare_models.py` — Comparative evaluation

### C. Experimental Configuration

| Parameter | Value |
|---|---|
| Network Nodes | 50 |
| Network Edges | 100 (bidirectional) |
| Topology Model | BRITE (Barabási-Albert) |
| Training Episodes | 3,000 |
| Random Seed | 42 |
| Max Steps/Episode | 500 |
| DT Grid Resolution | 41 × 41 |
| Jamming Radius | 10 DT units |
| Scenario Mix | 50% jammed + 50% normal |
| Anomaly Training Samples | 2,000 normal scenarios |
| Dataset Number | 0 |

---

## References

[1] S. S. Bhavanasi, L. Pappone, and F. Esposito, "Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning," submitted to IEEE TNSM, Special Issue on Reliable Networks, 2023.

[2] A. Krause, M. D. Khursheed, P. Schulz, F. Burmeister, and G. Fettweis, "Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks," Proc. 2023 IEEE Globecom Workshops, Kuala Lumpur, Malaysia, Dec. 2023.

[3] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015.

[4] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," Proc. ICLR, Toulon, France, Apr. 2017.

[5] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," Proc. ICLR, Vancouver, Canada, Apr. 2018.

[6] Z. Jin, Y. Wang, X. Li, and H. Chen, "Deep Reinforcement Learning-based Resilient Routing for Urban Emergency Communication Networks," IEEE Trans. Vehicular Technology, vol. 74, no. 1, pp. 15–28, 2025.

[7] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: MIT Press, 2018.

[8] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, "A Comprehensive Survey on Graph Neural Networks," IEEE Trans. NNLS, vol. 32, no. 1, pp. 4–24, Jan. 2021.

[9] M. M. Olsson, R. Barr, R. Hague, C. Sherburn, and J. Li, "Digital Twin Technology: An Overview," Annals of the CIRP, vol. 69, no. 1, pp. 629–642, 2020.

[10] L. Busoniu, R. Babuska, and B. De Schutter, "A Comprehensive Survey of Multiagent Reinforcement Learning," IEEE Trans. SMC-C, vol. 38, no. 2, pp. 156–172, Mar. 2008.

[11] I. F. Akyildiz, X. Wang, and W. Wang, "Wireless mesh networks: a survey," Computer Networks, vol. 47, no. 4, pp. 445–487, Mar. 2005.

[12] W. Xu, W. Trappe, Y. Zhang, and T. Wood, "The Feasibility of Launching and Detecting Jamming Attacks in Wireless Networks," Proc. ACM MobiHoc, Urbana-Champaign, IL, USA, May 2005, pp. 46–57.

[13] A. Medina, A. Lakhina, I. Matta, and J. Byers, "BRITE: An Approach to Universal Topology Generation," Proc. MASCOTS, Cincinnati, OH, USA, Aug. 2001.

[14] M. Fey and J. E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," ICLR Workshop, 2019.

[15] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, vol. 32, 2019.

[16] G. Brockman et al., "OpenAI Gym," arXiv preprint arXiv:1606.01540, 2016.

[17] H. Van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning with Double Q-learning," Proc. AAAI, vol. 30, no. 1, Phoenix, AZ, USA, Feb. 2016.

[18] A. Hagberg, P. Swart, and D. S Chult, "Exploring network structure, dynamics, and function using NetworkX," Proc. SciPy, Pasadena, CA, USA, 2008, pp. 11–15.

[19] Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," Proc. ICML, New York, NY, USA, Jun. 2016.

[20] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized Experience Replay," Proc. ICLR, San Juan, Puerto Rico, May 2016.

[21] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," Proc. ICLR, San Diego, CA, USA, May 2015.

[22] M. Grondman, L. Busoniu, G. A. D. Lopes, and R. Babuska, "A Survey of Actor-Critic Reinforcement Learning," IEEE Trans. SMC-C, vol. 42, no. 6, pp. 1291–1307, Nov. 2012.

[23] A. Albert and A. Barabási, "Statistical mechanics of complex networks," Reviews of Modern Physics, vol. 74, no. 1, pp. 47–97, Jan. 2002.

[24] C. Perkins, E. Belding-Royer, and S. Das, "Ad hoc On-Demand Distance Vector (AODV) Routing," IETF RFC 3561, Jul. 2003.

[25] T. Clausen and P. Jacquet, "Optimized Link State Routing Protocol (OLSR)," IETF RFC 3626, Oct. 2003.
