# Training and Execution Analysis

## Overview

This document provides a deep-dive analysis of the main execution scripts:
- `train_integrated.py`: Train the DT-enhanced GCN model
- `compare_models.py`: Compare GCN with/without DT features

---

## 1. `train_integrated.py`

### Purpose
Main training script for the Digital Twin + GCN integrated system.

### Usage
```bash
python train_integrated.py --episodes 1000 --save --plot
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | `int` | `500` | Number of training episodes |
| `--dataset_nr` | `int` | `0` | DT dataset number (0-5) |
| `--normal_samples` | `int` | `2000` | Normal samples for anomaly training |
| `--save` | `flag` | `False` | Save trained model weights |
| `--plot` | `flag` | `False` | Display training plots |

### Execution Flow

```python
def main(args):
    """
    Complete training pipeline:
    1. Load datasets
    2. Train anomaly detector
    3. Create environment
    4. Create model
    5. Train
    6. Save results
    """
```

### Step 1: Load DT Datasets

```python
# Paths
brite_path = os.path.join(os.path.dirname(__file__), "RP15", "50nodes.brite")
dataset_dir = os.path.join(os.path.dirname(__file__), "RP12_paper", "datasets")

# Initialize loader
loader = DTDatasetLoader(dataset_dir)

# Get scenario counts
num_scenarios = loader.get_scenario_count(args.dataset_nr)  # 30,000
num_jammed = len(loader.get_jammed_indices(args.dataset_nr))  # ~15,000
num_normal = len(loader.get_normal_indices(args.dataset_nr))  # ~15,000
```

**Output**:
```
Loading DT datasets...
  Total scenarios: 30000
  Jammed: 15000, Normal: 15000
```

### Step 2: Train Anomaly Detector

```python
anomaly_bridge = AnomalyBridge()
normal_measurements = loader.get_normal_measurements(args.dataset_nr)[:args.normal_samples]
anomaly_bridge.train(normal_measurements)
```

**Algorithm**:
1. Get first 2000 measurement diffs from non-jammed scenarios
2. Compute mean and std of absolute diffs
3. Set threshold = mean + 2×std

**Output**:
```
Training anomaly detector...
AnomalyBridge trained: mean=0.42, std=0.31, threshold=1.04
```

### Step 3: Create Hybrid Environment

```python
env = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
```

**Initialization**:
1. Load BRITE graph (50 nodes, 100 edges)
2. Scale node positions to DT coordinates (0-40)
3. Initialize MeshDigitalTwin and JammerDetector
4. Pre-load jammed/normal scenario indices
5. Build PyG edge_index tensor

**Output**:
```
Creating hybrid environment...
  Nodes: 50
  Node features: 7
```

### Step 4: Create GCN Model

```python
policy_net = GCN_DTAware(
    num_nodes=env.num_nodes,    # 50
    out_dim=env.num_nodes,      # 50
    node_feat_dim=NODE_FEATURE_DIM  # 7
)
target_net = GCN_DTAware(...)

num_params = sum(p.numel() for p in policy_net.parameters())
```

**Architecture Details**:
```
GATConv(7 → 16): ~16 × 7 + 16 = 128 params
GATConv(16 → 25): ~25 × 16 + 25 = 425 params
Linear(1250 → 50): 1250 × 50 + 50 = 62,550 params
Linear(50 → 50): 50 × 50 + 50 = 2,550 params
Total: ~65,653 parameters
```

**Output**:
```
Creating GCN_DTAware model...
  Parameters: 65,653
```

### Step 5: Create Agent and Train

```python
agent = GCN_DTAgent(
    num_nodes=env.num_nodes,
    policy_net=policy_net,
    target_net=target_net,
    env=env
)

agent.run(num_episodes=args.episodes, verbose=True)
```

**Training Loop**:
```
For episode in range(num_episodes):
    # Reset with random scenario
    obs, info = env.reset()
    
    For step in count():
        # ε-greedy action selection
        action = agent.select_action(obs, valid_actions)
        
        # Execute and get reward
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        agent.memory.push(state, action, next_state, reward)
        
        # Optimize every 4 steps
        if step_count % 4 == 0:
            agent.optimize_model()
        
        # Update target network every 14000 steps
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            record_episode_metrics()
            break
```

**Output** (verbose):
```
Training: 100%|██████████| 500/500 [12:34<00:00, 1.51s/it]

Episode 50: avg_reward=-0.23, success_rate=32.00%
Episode 100: avg_reward=0.15, success_rate=48.00%
Episode 150: avg_reward=0.42, success_rate=61.00%
...
Episode 500: avg_reward=0.78, success_rate=85.00%

Training complete. Final success rate: 85.00%
```

### Step 6: Save Metrics to CSV

```python
save_metrics_to_csv(agent.metrics, args.episodes, elapsed)
```

**Generated Files**:

1. **Episode Metrics** (`episode_metrics_{timestamp}.csv`):
```csv
episode,episode_reward,path_length,jammed_steps,total_latency,avg_bandwidth,avg_pdr
1,-0.51,8,2,0.45,0.67,0.67
2,0.75,5,0,0.32,0.81,0.81
...
```

2. **Step Metrics** (`step_metrics_{timestamp}.csv`):
```csv
step,loss,latency,bandwidth
1,0.234,0.12,0.78
2,0.198,0.15,0.65
...
```

3. **Training Summary** (`training_summary.csv`):
```csv
timestamp,model,episodes,training_time_min,avg_reward,max_reward,min_reward,success_rate,avg_path_length,avg_jammed_steps,avg_latency,avg_bandwidth,avg_pdr,final_50_avg_reward
20260203_190000,GCN_DTAware,500,12.5,0.42,1.31,-1.51,0.72,6.3,0.8,0.28,0.65,0.65,0.78
```

### Step 7: Save Model (Optional)

```python
if args.save:
    save_path = os.path.join(..., "integrated_dt_gcn", "trained_model.pt")
    agent.save(save_path)
```

**Saved Content**:
```python
{
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'steps_done': agent.steps_done
}
```

### Step 8: Plot Metrics (Optional)

```python
if args.plot:
    plot_metrics(agent.metrics)
```

**Generated Plots** (2×2 grid):
- Episode Reward (with MA-20)
- Training Loss
- Path Length
- Jammed Steps per Episode

---

## 2. `compare_models.py`

### Purpose
Compare GCN performance with and without Digital Twin features.

### Usage
```bash
python compare_models.py --episodes 300 --plot
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | `int` | `300` | Episodes per model |
| `--dataset_nr` | `int` | `0` | DT dataset number |
| `--plot` | `flag` | `True` | Generate comparison plots |

### Execution Flow

### Step 1: Setup (Same as train_integrated.py)

```python
# Load datasets and train anomaly detector
loader = DTDatasetLoader(dataset_dir)
anomaly_bridge = AnomalyBridge()
normal_measurements = loader.get_normal_measurements(args.dataset_nr)[:2000]
anomaly_bridge.train(normal_measurements)
```

### Step 2: Train Model 1 - GCN with DT (7 features)

```python
env_dt = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)
result_dt = train_model(env_dt, NUM_NODES, NODE_FEATURE_DIM, 
                        args.episodes, "GCN + DT (7 features)")
```

This uses the full 7-dimensional features:
- `[0-3]`: Original routing features
- `[4]`: Anomaly score
- `[5]`: Jam probability
- `[6]`: Neighbor jam average

### Step 3: Train Model 2 - GCN Baseline (4 features)

```python
env_baseline = HybridEnv(brite_path, loader, anomaly_bridge, args.dataset_nr)

# Monkey-patch to disable DT features
original_compute = env_baseline._compute_node_features
def compute_4_features(self=env_baseline):
    x = original_compute()
    x[:, 4:] = 0  # Zero out DT features
    return x
env_baseline._compute_node_features = compute_4_features

result_baseline = train_model(env_baseline, NUM_NODES, NODE_FEATURE_DIM,
                              args.episodes, "GCN Baseline (4 features)")
```

**Key Insight**: Both models use 7-dimensional input, but baseline has zeros for DT features. This ensures fair comparison (same model architecture).

### Step 4: Compare Results

```python
# Print comparison table
print(f"{'Metric':<25} {'GCN + DT':<15} {'GCN Baseline':<15}")
print("-" * 55)
print(f"{'Avg Reward':<25} {result_dt['avg_reward']:<15.3f} {result_baseline['avg_reward']:<15.3f}")
print(f"{'Success Rate':<25} {result_dt['success_rate']:<15.2%} {result_baseline['success_rate']:<15.2%}")
print(f"{'Avg Path Length':<25} {result_dt['avg_path_length']:<15.2f} {result_baseline['avg_path_length']:<15.2f}")
print(f"{'Avg Jammed Steps':<25} {result_dt['avg_jammed_steps']:<15.2f} {result_baseline['avg_jammed_steps']:<15.2f}")
```

**Example Output**:
```
==============================================================
COMPARISON RESULTS
==============================================================
Metric                    GCN + DT        GCN Baseline   
-------------------------------------------------------
Avg Reward                0.685           0.423          
Success Rate              82.00%          65.00%         
Avg Path Length           5.80            7.20           
Avg Jammed Steps          0.45            1.85           
Training Time (min)       8.2             7.9            
==============================================================
```

### Step 5: Save Comparison CSV

**Generated Files**:

1. **Comparison Episodes** (`comparison_episodes_{timestamp}.csv`):
```csv
episode,GCN_+_DT_7_features_reward,GCN_Baseline_4_features_reward
1,-0.32,0.15
2,0.78,-0.45
...
```

2. **Comparison Summary** (`comparison_summary_{timestamp}.csv`):
```csv
timestamp,model,episodes,training_time_min,avg_reward,success_rate,avg_path_length,avg_jammed_steps,avg_pdr
...,GCN + DT (7 features),300,8.2,0.685,0.82,5.8,0.45,0.72
...,GCN Baseline (4 features),300,7.9,0.423,0.65,7.2,1.85,0.68
```

3. **Improvement Metrics** (`comparison_improvements.csv`):
```csv
timestamp,dt_model,baseline_model,reward_improvement,success_rate_improvement_pp,jammed_steps_reduction,path_length_reduction,pdr_improvement
...,GCN + DT (7 features),GCN Baseline (4 features),0.262,17.00,1.40,1.40,0.04
```

### Step 6: Generate Comparison Plots

```python
if args.plot:
    plot_comparison(results)
```

**Generated Plot** (`comparison_results.png`):

```
┌─────────────────────────────────────────────────────────────┐
│  Subplot 1: Episode Reward (Moving Average)                 │
│  - Green line: GCN + DT                                     │
│  - Blue line: GCN Baseline                                  │
├─────────────────────────────────────────────────────────────┤
│  Subplot 2: Bar Chart Comparison                            │
│  - Success Rate, Avg Reward, Jammed Steps                   │
│  - Side-by-side bars for each model                         │
├─────────────────────────────────────────────────────────────┤
│  Subplot 3: Training Loss                                   │
│  - Smoothed loss curves for both models                     │
├─────────────────────────────────────────────────────────────┤
│  Subplot 4: Summary Text Box                                │
│  - Key statistics and improvements                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Helper Function: `train_model()`

```python
def train_model(env, num_nodes: int, node_feat_dim: int, 
                num_episodes: int, name: str) -> dict:
    """Train a model and return metrics."""
    
    # Create networks
    policy_net = GCN_DTAware(num_nodes, num_nodes, node_feat_dim)
    target_net = GCN_DTAware(num_nodes, num_nodes, node_feat_dim)
    
    # Create agent
    agent = GCN_DTAgent(num_nodes, policy_net, target_net, env)
    
    # Train
    start_time = time.time()
    agent.run(num_episodes, verbose=True)
    elapsed = time.time() - start_time
    
    # Calculate metrics from last 100 episodes
    final_rewards = agent.metrics['eps_reward'][-100:]
    final_path_lengths = agent.metrics['path_length'][-100:]
    final_jammed = agent.metrics['jammed_steps'][-100:]
    
    successes = sum(1 for r in final_rewards if r > 0)
    
    return {
        'name': name,
        'time': elapsed,
        'avg_reward': np.mean(final_rewards),
        'avg_path_length': np.mean(final_path_lengths),
        'avg_jammed_steps': np.mean(final_jammed),
        'success_rate': successes / len(final_rewards),
        'all_rewards': agent.metrics['eps_reward'],
        'all_losses': agent.metrics['loss'],
        'avg_pdr': np.mean(agent.metrics.get('eps_avg_pdr', [0])),
        'all_pdrs': agent.metrics.get('eps_avg_pdr', [])
    }
```

---

## 4. Metrics Explanation

### Per-Episode Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `episode_reward` | Sum of rewards in episode | Higher = better routing |
| `path_length` | Number of hops taken | Lower = more efficient |
| `jammed_steps` | Number of jammed nodes visited | Lower = better avoidance |
| `total_latency` | Sum of edge weights on path | Lower = faster path |
| `avg_bandwidth` | Mean edge capacity on path | Higher = more capacity |
| `avg_pdr` | Mean packet delivery ratio | Higher = more reliable |

### Per-Step Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Huber loss from DQN optimization |
| `latency` | Edge weight of current hop |
| `bandwidth` | Edge capacity of current hop |

### Summary Metrics

| Metric | Description |
|--------|-------------|
| `success_rate` | % of episodes with positive final reward |
| `final_50_avg_reward` | Mean reward of last 50 episodes |

---

## 5. Expected Output Files

After running `train_integrated.py`:
```
integrated_dt_gcn/results/
├── episode_metrics_20260203_190000.csv
├── step_metrics_20260203_190000.csv
├── training_summary.csv (appended)
└── training_metrics.png

integrated_dt_gcn/
└── trained_model.pt (if --save)
```

After running `compare_models.py`:
```
integrated_dt_gcn/results/
├── comparison_episodes_20260203_200000.csv
├── comparison_summary_20260203_200000.csv
├── comparison_improvements.csv (appended)
└── comparison_results.png
```

---

## 6. Expected Performance

### Typical Results (500 episodes)

| Model | Success Rate | Avg Reward | Jammed Steps |
|-------|--------------|------------|--------------|
| GCN + DT | 80-90% | 0.6-0.9 | 0.3-0.8 |
| GCN Baseline | 60-75% | 0.3-0.5 | 1.5-2.5 |

### Key Improvements from DT Integration

| Metric | Improvement |
|--------|-------------|
| Success Rate | +15-25 percentage points |
| Jammed Steps | -1.0 to -2.0 steps |
| Average Reward | +0.2 to +0.4 |

The DT integration enables the model to **proactively avoid jammed nodes** by using anomaly scores and jam probabilities as additional features.
