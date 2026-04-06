# Chapter 2: Related Works

This chapter reviews the existing literature across four key domains that underpin the proposed framework: reinforcement learning for network routing, graph neural networks for networking applications, Digital Twin technologies for radio environments, and multi-agent systems in network optimization. A gap analysis at the end identifies the specific contributions of this work.

## 2.1 Reinforcement Learning in Network Routing

Routing in Wireless Mesh Networks is typically modeled as a Markov Decision Process (MDP), which provides a mathematical framework for decision-making in stochastic environments. In this formulation, the network state includes information about topology, link quality, and traffic conditions; actions correspond to routing decisions (selecting the next hop for a packet); and rewards reflect the quality of the routing decision (e.g., successful delivery, low latency, avoiding jammed areas).

Reinforcement Learning (RL) techniques have shown considerable promise in adaptive, real-time decision-making in dynamic environments. Unlike traditional optimization methods that require complete knowledge of the environment, RL agents learn optimal policies through interaction with the environment, making them well-suited for dynamic network conditions where the environment model is unknown or constantly changing. Sutton and Barto (2018) provide a comprehensive treatment of RL theory and algorithms in their seminal textbook "Reinforcement Learning: An Introduction."

### 2.1.1 Q-Learning and Deep Q-Networks

Q-learning, introduced by Watkins and Dayan (1992), is a model-free RL algorithm that learns the value of taking a specific action in a given state. The Q-value function Q(s, a) represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter. The Q-value update rule is:

> Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') − Q(s, a)]

where α is the learning rate, γ is the discount factor, r is the immediate reward, and s' is the next state. Q-learning converges to the optimal Q-function under certain conditions (all state-action pairs visited infinitely often, learning rate decay), but maintaining a Q-table becomes infeasible for large or continuous state spaces.

Deep Q-Networks (DQN), introduced by Mnih et al. (2015) in their landmark paper "Human-level control through deep reinforcement learning," extend Q-learning by using deep neural networks to approximate the Q-value function, enabling the algorithm to handle high-dimensional state spaces. DQN introduced two key innovations that significantly improve training stability:

1. **Experience Replay:** Transitions (s, a, r, s') are stored in a replay buffer and sampled randomly for training. This breaks the temporal correlation between consecutive samples, reducing variance and improving data efficiency. The replay buffer acts as a form of data augmentation, allowing each experience to be used multiple times during training.

2. **Target Network:** A separate neural network, updated periodically, is used for computing target Q-values. This reduces the moving target problem where the network is trained to approximate values that are themselves changing. The target network provides more stable training targets, leading to smoother and more reliable convergence.

Van Hasselt et al. (2016) proposed Double DQN (DDQN) to address the overestimation bias inherent in standard DQN. In standard DQN, the same network is used both to select and evaluate actions, leading to systematic overestimation of Q-values. DDQN decouples action selection from action evaluation by using the policy network to select the best action but the target network to evaluate its Q-value:

> Y_DDQN = r + γ Q_target(s', argmax_a Q_policy(s', a))

Wang et al. (2016) introduced Dueling DQN, which separates the Q-function into a state-value function V(s) and an advantage function A(s, a), allowing the network to learn which states are valuable independent of the action taken. This architecture is particularly beneficial when many actions have similar values.

Schaul et al. (2016) proposed Prioritized Experience Replay, which samples experiences from the replay buffer based on their temporal-difference error rather than uniformly. Experiences with larger errors are sampled more frequently, focusing training on the most informative transitions and improving sample efficiency.

### 2.1.2 Deep Reinforcement Learning for Network Routing

The application of deep reinforcement learning to network routing has gained significant attention in recent years. Several key works have demonstrated the potential of DRL for adaptive, intelligent routing.

Bhavanasi, Pappone, and Esposito (2023) proposed a framework in their paper "Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning" that integrates Graph Neural Networks with Multi-Agent Deep Reinforcement Learning (DRL) to overcome a core limitation of traditional RL: the inability to generalize to unfamiliar network topologies without significant retraining. By encoding the network state using GNNs, their model captures the graph's permutation-invariant structure, enabling agents to minimize flow collisions and adapt to real-time topology changes dynamically. Their MA-DQN model implements one DQN agent per node in a 50-node Barabási-Albert network, with each agent learning to forward packets to the optimal neighbour. Their work demonstrated that GNN-based state representations enable RL agents to transfer learned policies across different network topologies, significantly reducing the need for retraining. The multi-agent architecture proved more effective than single-agent approaches for distributed routing, as each node can make locally optimal decisions that collectively produce globally effective routing. This work provides the foundational multi-agent architecture for the framework proposed in this dissertation.

Jin et al. (2025) researched a Deep Reinforcement Learning-based Resilient Routing (DRLRR) approach for urban emergency communication networks in their paper "Deep Reinforcement Learning-based Resilient Routing for Urban Emergency Communication Networks." To counter the overestimation bias of conventional Q-learning, they used Double Deep Q-Networks (DDQN), enabling stable route optimization despite excessive node degradation scenarios. Their approach integrates node and link state information into a specialized reward function, ensuring stable data transmission during disasters. Their work highlights the importance of resilient routing in adversarial conditions but uses a single-agent architecture and does not leverage graph structure.

Almasan et al. (2022) proposed RouteNet-Fermi, a GNN-based model that achieves unprecedented accuracy in predicting network performance metrics such as delay, jitter, and packet loss. While RouteNet-Fermi is primarily a performance prediction model rather than a routing agent, its success demonstrates the power of GNNs for understanding network behaviour and motivates their use in routing decision-making.

Stampa et al. (2017) were among the earliest to apply deep reinforcement learning to network routing, demonstrating that DRL agents can learn routing policies that outperform shortest-path algorithms in terms of load balancing and congestion avoidance. Their work showed that RL agents can discover non-obvious routing strategies that traditional algorithms miss.

### 2.1.3 Actor-Critic and Policy Gradient Methods

While DQN and its variants learn a value function and derive a policy implicitly, actor-critic methods maintain both an explicit policy (the actor) and a value function (the critic). The actor selects actions based on the current policy, while the critic evaluates the actions by estimating the value function. This dual-network architecture enables more stable learning in continuous action spaces and high-dimensional environments.

Grondman et al. (2012) provide a comprehensive survey of actor-critic reinforcement learning methods in their paper "A Survey of Actor-Critic Reinforcement Learning." They categorize actor-critic methods based on the type of critic (state-value, action-value, advantage) and the policy parameterization (stochastic, deterministic).

Proximal Policy Optimization (PPO), introduced by Schulman et al. (2017), has become one of the most widely used policy gradient methods due to its simplicity and reliability. PPO uses a clipped surrogate objective to prevent excessively large policy updates, ensuring stable and monotonic improvement. While PPO has been applied to various networking problems, its continuous-action formulation is less natural for discrete routing decisions compared to DQN-based approaches.

For the routing problem considered in this dissertation, DQN-based methods are preferred over actor-critic methods because: (1) routing actions are inherently discrete (selecting a neighbour), (2) DQN's replay buffer enables better sample efficiency for the multi-agent setting, and (3) the Q-value formulation naturally maps to the next-hop selection problem.

## 2.2 Graph Neural Networks for Networking

Graph Neural Networks (GNNs) have emerged as a powerful tool for learning on graph-structured data, making them naturally suited for networking applications where the network topology can be represented as a graph. GNNs operate by iteratively aggregating information from neighbouring nodes, allowing each node to build a representation that captures both its local features and its structural context within the graph. Wu et al. (2021) provide a comprehensive survey of GNN architectures and applications in their paper "A Comprehensive Survey on Graph Neural Networks."

### 2.2.1 Message Passing Neural Networks

The Message Passing Neural Network (MPNN) framework, introduced by Gilmer et al. (2017), provides a unifying framework for understanding GNN architectures. In the MPNN framework, each node maintains a hidden state that is iteratively updated through two phases:

1. **Message Phase:** Each node collects messages from its neighbours. The message function takes the hidden states of the sending and receiving nodes, along with edge features, and produces a message vector.

2. **Update Phase:** Each node updates its hidden state based on the aggregated messages from its neighbours. The update function combines the node's current state with the aggregated messages to produce a new hidden state.

This framework encompasses most popular GNN architectures, including GCN, GAT, GraphSAGE, and GIN. The key differences between these architectures lie in their message and aggregation functions.

### 2.2.2 Graph Convolutional Networks

Graph Convolutional Networks (GCNs), introduced by Kipf and Welling (2017) in their paper "Semi-Supervised Classification with Graph Convolutional Networks," gained popularity as an efficient spectral approximation of convolution on graphs. Each node's representation is iteratively refined by aggregating the features of its local neighbours, normalized by the degrees of related nodes. The GCN layer-wise propagation rule is:

> H^(l+1) = σ(D̃^(−½) Ã D̃^(−½) H^(l) W^(l))

where Ã = A + I_N is the adjacency matrix with added self-connections, D̃ is the degree matrix of Ã, W^(l) is the layer-specific trainable weight matrix, and σ is the activation function. During convolution, the GCN assumes a constant graph structure and uses only the adjacency of the graph to determine aggregation weights instead of learned pairwise significance scores. While computationally efficient, this uniform weighting scheme limits the model's ability to distinguish between more and less important neighbours — a critical limitation for routing in adversarial environments where link quality varies significantly.

### 2.2.3 Graph Attention Networks

Graph Attention Networks (GATs), introduced by Veličković et al. (2018) in their paper "Graph Attention Networks," address the limitations of GCNs by introducing a self-attention mechanism that computes attention coefficients α_ij indicating how important neighbour j is to node i:

> α_ij = exp(LeakyReLU(a^T[Wh_i ∥ Wh_j])) / Σ_k∈N_i exp(LeakyReLU(a^T[Wh_i ∥ Wh_k]))

Unlike GCN, which assigns uniform weights during aggregation, GAT selectively weights each neighbour's contribution through a learnable attention mechanism. This makes it particularly suited for environments where link reliability varies due to jamming or interference. Selective aggregation allows GAT to better capture the importance of high-quality, non-jammed links during routing, which is a core motivation for its adoption in this framework.

GATs also support multi-head attention, where K independent attention mechanisms are applied and their outputs are concatenated or averaged:

> h_i' = ∥_{k=1}^{K} σ(Σ_{j∈N_i} α_ij^k W^k h_j)

This stabilizes the learning process and allows the model to attend to information from different representation subspaces at different positions. Multi-head attention increases the expressiveness of the model without significantly increasing computational cost.

The PyTorch Geometric (PyG) library, developed by Fey and Lenssen (2019) and described in their paper "Fast Graph Representation Learning with PyTorch Geometric," provides efficient implementations of GATConv layers with support for edge attributes. This edge attribute support is particularly important for the proposed framework, as it allows Digital Twin-derived link quality metrics to influence the attention computation.

### 2.2.4 GNNs for Network Performance

The application of GNNs to networking problems has yielded impressive results across several domains:

- **Traffic Engineering:** GNNs have been used to predict network traffic patterns and optimize routing to minimize congestion. The graph structure of the network naturally maps to the input requirements of GNNs.
- **Network Planning:** GNN-based models can predict the performance impact of topology changes, enabling more informed network planning decisions.
- **Anomaly Detection:** GNNs can learn normal communication patterns and detect deviations that may indicate attacks or failures.
- **Resource Allocation:** In wireless networks, GNNs have been applied to interference management and resource allocation problems.

## 2.3 Digital Twin for Radio Environments

The concept of a Digital Twin — a real-time virtual replica of a physical system — has gained significant traction in the context of wireless network management. Originally coined by Grieves in 2002 for manufacturing applications, the Digital Twin paradigm has evolved to encompass any system where a virtual model is continuously synchronized with its physical counterpart. Olsson et al. (2020) provide an overview of Digital Twin technology and its applications across various domains in their paper "Digital Twin Technology: An Overview."

### 2.3.1 Digital Twin Architecture

A Digital Twin for radio environments typically consists of several interconnected components:

- **Physical Twin:** The actual wireless network with its nodes, links, and radio environment. Sensors and measurement equipment capture real-time data about the network state.
- **Virtual Twin:** A computational model that mirrors the physical network's behaviour. The virtual twin maintains representations of the network topology, radio propagation characteristics, and environmental conditions.
- **Data Connection:** A bidirectional data flow between the physical and virtual twins. Real-time measurements from the physical network update the virtual model, while the virtual model's predictions and analyses inform decisions in the physical network.
- **Radio Map:** A spatial representation of the radio environment containing Received Signal Strength (RSS) values at grid points across the coverage area. Radio maps capture the effects of path loss, shadowing, and interference.
- **Path Loss Model:** A mathematical model that predicts signal attenuation based on distance and environmental factors. Common models include the Free-Space Path Loss (FSPL) model, the log-distance path loss model, and ray-tracing approaches.
- **Anomaly Detection:** Modules that compare the expected radio environment (predicted by the virtual twin) with actual measurements to identify deviations indicating jammers or other anomalies.

### 2.3.2 Digital Twin for Anomaly Detection

Krause et al. (2023) proposed a novel approach to anomaly detection in radio environments using Digital Twins in their paper "Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks," published in the Proceedings of the 2023 IEEE Globecom Workshops. Rather than anchoring detection to patterns mined from past data, their system maintains a physics-based virtual replica of the network running in real time. A ray-tracing engine inside the replica computes expected RSSI readings across each link under normal conditions. Any meaningful gap between predicted and measured signals is flagged — no attack examples are needed for training.

Crucially, their framework can distinguish jamming from ordinary fading, which is exactly where conventional statistical detection tends to fail. By comparing the Digital Twin's predicted radio environment with actual measurements, the system can identify anomalies attributable to intentional interference rather than natural signal fluctuations. The dataset generated by Krause et al. forms the basis for the Digital Twin component in the proposed framework. It provides 30,000 radio map scenarios per dataset, including both normal and jammed conditions at a 41×41 grid resolution.

### 2.3.3 Digital Twin in IoT and Industry 4.0

Beyond anomaly detection, Digital Twins have found applications in various networking contexts:

- **Network Planning and Optimization:** Digital Twins enable what-if analysis for network planning decisions, allowing operators to evaluate the impact of topology changes, capacity upgrades, or interference mitigation strategies in a virtual environment before implementing them in the physical network.
- **Predictive Maintenance:** By continuously monitoring the gap between expected and actual network behaviour, Digital Twins can predict impending failures and trigger proactive maintenance actions.
- **Security Analysis:** Digital Twins provide a sandbox environment for testing security policies, simulating attack scenarios, and evaluating the effectiveness of countermeasures without risking the physical network.

## 2.4 Multi-Agent Systems in Network Optimization

Multi-Agent Reinforcement Learning (MARL) extends traditional RL by enabling multiple agents to learn and make decisions simultaneously within a shared environment. In the context of network routing, MARL is a natural fit because routing decisions are inherently decentralized — each node in the network independently decides how to forward packets based on local information. Busoniu et al. (2008) provide a comprehensive survey of multi-agent reinforcement learning in their paper "A Comprehensive Survey of Multiagent Reinforcement Learning."

### 2.4.1 MARL Paradigms

Several paradigms exist for organizing multi-agent learning:

- **Independent Learners (IL):** Each agent learns independently, treating other agents as part of the environment. While simple, this approach can suffer from non-stationarity as the environment changes from each agent's perspective due to the evolving policies of other agents.
- **Centralized Training with Decentralized Execution (CTDE):** Agents have access to global information during training but must act based on local observations during execution. This paradigm balances the benefits of centralized learning with the practical requirements of decentralized deployment.
- **Fully Centralized:** A single controller makes decisions for all agents. While this enables optimal coordination, it introduces a single point of failure and may not scale to large networks.
- **Communication-Based:** Agents can exchange messages to coordinate their actions. The communication protocol can be learned alongside the policy, enabling agents to develop task-specific communication strategies.

### 2.4.2 Cooperative Multi-Agent Learning

Cooperative MARL introduces mechanisms for agents to share information and coordinate their learning. Common approaches include:

- **Shared Experience Replay:** Agents contribute their experiences to a shared replay buffer, enabling each agent to learn from the experiences of all agents. This is particularly effective when agents face similar decision-making challenges.
- **Parameter Sharing:** All agents share the same neural network parameters, reducing the total number of parameters and enabling faster learning. However, parameter sharing requires agents to have the same action space, which may not hold in heterogeneous networks.
- **Reward Shaping:** Agents receive a combination of local and global rewards, encouraging both individually optimal and collectively beneficial behaviour.

The work by Bhavanasi, Pappone, and Esposito (2023) in "Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning" provides the foundational architecture for the multi-agent component of this framework. Their MA-DQN model implements one DQN agent per node in a 50-node Barabási-Albert network. Each agent has its own policy network with output dimension equal to the number of neighbours, enabling locally optimal next-hop decisions. This design ensures that the multi-agent system respects the decentralized nature of mesh routing.

## 2.5 Gap Analysis

The review of existing literature reveals several significant gaps that motivate the proposed framework:

**Gap 1: Limited Integration of Digital Twin and Routing.** Bhavanasi, Pappone, and Esposito (2023) built a solid routing foundation using GNNs and multi-agent DRL, yet the work overlooks the subtleties of physical-layer threats by anchoring itself to standard network metrics. The agents have no awareness of the underlying physical layer conditions, meaning they cannot proactively avoid jammed areas.

**Gap 2: Detection Without Intervention.** Krause et al. (2023) leverage digital twins for detection but deliberately stop at observation, leaving the door closed on any form of active network intervention. Their framework identifies anomalies but does not take any corrective action.

**Gap 3: Missing Feature Enrichment.** Existing work has yet to meaningfully explore the tight integration of Digital Twin anomaly detection and GNN-based routing — particularly feeding high-fidelity anomaly scores from a Digital Twin directly into a Graph Attention Network as live, dynamic features.

**Gap 4: Lack of Attention-Based Link Assessment.** Most GNN-based routing approaches use GCN layers that assign uniform weights to all neighbours during aggregation. In adversarial environments where link quality varies significantly, attention mechanisms can provide more effective routing decisions by selectively focusing on high-quality links.

**Gap 5: Single-Agent Limitations.** Many DRL-based routing approaches use a single centralized agent, which contradicts the distributed nature of mesh networking and introduces scalability limitations.

This dissertation addresses these gaps by integrating all three capabilities: DT-based anomaly detection, GAT-based topology embedding with DT-enriched features, and multi-agent DRL for distributed routing.

### Comparative Summary of Related Works

| Reference | Approach | Strengths | Limitations |
|---|---|---|---|
| Bhavanasi, Pappone, and Esposito (2023) | GNN + Multi-Agent DRL | Topology-aware routing, generalization across topologies | No physical-layer threat awareness, no DT integration |
| Jin et al. (2025) | DDQN for Emergency Routing | Resilient routing under node degradation | Single-agent, no graph structure utilization |
| Kipf and Welling (2017) | GCN | Efficient spectral graph learning | Uniform neighbour weighting |
| Veličković et al. (2018) | GAT | Selective neighbour weighting via attention | Not designed for routing applications |
| Krause et al. (2023) | DT Anomaly Detection | Physics-based jamming detection | Detection only — no active routing response |
| Mnih et al. (2015) | DQN | Handles high-dimensional state spaces | Single-agent, no graph awareness |
| Van Hasselt et al. (2016) | Double DQN | Reduces overestimation bias | Single-agent architecture |
| Almasan et al. (2022) | RouteNet-Fermi (GNN) | Accurate performance prediction | Prediction only, not a routing agent |
| **This Work** | **DT + GAT + Multi-Agent DQN** | **Integrates detection, representation, and distributed routing** | **Simulation-based evaluation only** |

---
