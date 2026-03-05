# Paper Edit Guide — Align Text with Implementation

> Only **necessary** edits listed. Style and tone are preserved.  
> Changes marked as: ~~strikethrough~~ = remove, **bold** = add/replace

---

## Which File to Edit

**File:** `Resilient Routing in Wireless Mesh Networks using Digital Twin and GNN.docx`  

The Related works file (`Related works1.docx`) uses "MA DT-GCN" as a model name throughout (which is fine — it's a label, not a technical description), and its technical claims already align with the code. **No edits are needed in Related works1.docx.**

---

## Edits for: `Resilient Routing in Wireless Mesh Networks using Digital Twin and GNN.docx`

### Edit 0 — Abstract: GCN reference (Paragraph 3)

The abstract says "GCN model" but the implementation uses GAT.

**Current text:**
> ...while a GCN model captures the topological and traffic features.

**Replace with:**
> ...while a GAT model captures the topological and traffic features.

---

### Edit 1 — Introduction: GCN description (Paragraph 14)

This paragraph describes plain spectral GCN, but the implementation uses GAT (attention-based). The methods section (P33) already correctly says "Graph Attention Network layers". This intro paragraph should match.

**Current text:**
> Graph Convolutional Network (GCN) is a spectral-based Graph Neural Network (GNN) framework. They are used to extract structural features from the network graph. By aggregating this information through efficient layer-wise propagation rules, GCNs can generate node embeddings which helps in guiding routing decisions. This can also facilitate the detection of anomalies by understanding the global topology.

**Replace with:**
> Graph Attention Network (GAT) is an attention-based Graph Neural Network (GNN) framework. They are used to extract structural features from the network graph. By aggregating information from neighbouring nodes through learned attention weights, GATs can generate node embeddings which helps in guiding routing decisions. This can also facilitate the detection of anomalies by understanding the global topology.

---

### Edit 2 — Introduction: Partial observability claim (Paragraph 13)

The code gives each agent the **full global graph** (not a local neighborhood view). Soften "partial observability" to match.

**Current text:**
> Each node has an agent embedded with it. The agent learns to optimize local and global performance under partial observability.

**Replace with:**
> Each node has an agent embedded with it. The agent learns to optimize local and global performance using its own policy network while sharing experiences across agents.

---

### Edit 3 — Methods: GCN section heading and description (Paragraph 30–31)

The heading says "Graph Convolutional Networks" but the method uses GAT. Align heading and text.

**Current heading (P30):**
> Graph Convolutional Networks

**Replace heading with:**
> Graph Attention Networks

**Current text (P31):**
> For the proposed framework, the decision making is implemented using a Graph Convolutional Network (GCN). It is used to create an embedding of the entire topology of the network and the dependencies between each node.

**Replace with:**
> For the proposed framework, the decision making is implemented using a Graph Attention Network (GAT). It is used to create an embedding of the entire topology of the network and the dependencies between each node.

---

### Edit 4 — Methods: Input representation (Paragraph 32)

Two references to "GCN" should say "GAT".

**Current text:**
> The seven-dimensional feature vector that was referred to earlier in Section 3.1 is given as the input to the GCN. Graph-based formulation of GCN helps the model to learn about both the topography and environmental information.

**Replace with:**
> The seven-dimensional feature vector that was referred to earlier in Section 3.1 is given as the input to the GAT. Graph-based formulation of GAT helps the model to learn about both the topography and environmental information.

---

### Edit 5 — Methods: DQN Learning (Paragraph 36)

One reference to "GCN's" should say "GAT's".

**Current text:**
> The agent combines the GCN's state representations with Reinforcement Learning techniques.

**Replace with:**
> The agent combines the GAT's state representations with Reinforcement Learning techniques.

---

### Edit 6 — Introduction: Technology integration (Paragraph 10)

Align the abbreviation with the actual architecture.

**Current text:**
> ...this paper proposed a novel approach that integrates three technologies including Digital Twin (DT), Multi-Agent Deep Q-Networks (DQN) and GCN.

**Replace with:**
> ...this paper proposed a novel approach that integrates three technologies including Digital Twin (DT), Multi-Agent Deep Q-Networks (DQN) and GAT.

---

## Summary of All Edits

| # | Paragraph | What Changes | Why |
|---|---|---|---|
| 0 | P3 (Abstract) | "a GCN model" → "a GAT model" | Code uses `GATConv`, not spectral GCN |
| 1 | P14 (Intro) | "GCN is a spectral-based..." → "GAT is an attention-based..." | Code uses `GATConv`, not spectral GCN |
| 2 | P13 (Intro) | "under partial observability" → "using its own policy network while sharing experiences" | Code uses full global graph, not partial obs |
| 3 | P30 (Methods heading) | "Graph Convolutional Networks" → "Graph Attention Networks" | Align heading with actual architecture |
| 4 | P31 (Methods) | "using a Graph Convolutional Network (GCN)" → "using a Graph Attention Network (GAT)" | Same — align terminology |
| 5 | P32 (Methods) | "input to the GCN. Graph-based formulation of GCN" → "input to the GAT. Graph-based formulation of GAT" | Same |
| 6 | P36 (Methods) | "GCN's state representations" → "GAT's state representations" | Same |
| 7 | P10 (Intro) | "...and GCN" → "...and GAT" | Same |

> [!NOTE]
> The model name **"MA DT-GCN"** used throughout both papers can stay as-is — it is a project/protocol label, not a technical description. The methods section correctly describes the actual GAT layers used.

> [!IMPORTANT]
> These edits cover **only text alignment**. No changes to experimental results, numbers, or conclusions are needed — those already match the code's metrics tracking.
