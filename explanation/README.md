# Repository Documentation

## Documentation Suite

This folder contains comprehensive documentation for the **Digital Twin-Enhanced GNN Routing** research repository.

---

## Quick Navigation

| # | Document | Description |
|---|----------|-------------|
| 1 | [System Overview](01_overview.md) | Architecture, data flow, folder structure |
| 2 | [Integrated DT-GCN Module](02_integrated_dt_gcn.md) | Core integration layer deep dive |
| 3 | [RP15 Module](03_RP15_module.md) | Baseline GNN routing module |
| 4 | [RP12 Paper Module](04_RP12_module.md) | Digital Twin & datasets |
| 5 | [Training & Execution](05_training_and_execution.md) | Training scripts analysis |
| 6 | [Variables Reference](06_variables_reference.md) | All constants and parameters |
| 7 | [DT Integration Deep Dive](07_digital_twin_integration.md) | How DT signals enable routing |
| 8 | [Dataset Documentation](08_datasets.md) | Dataset files, formats, loading |
| 9 | [BRITE Topology](09_brite_topology.md) | Network topology file format |

---

## Reading Order

### For New Users
1. **[01_overview.md](01_overview.md)** - Understand the system architecture
2. **[05_training_and_execution.md](05_training_and_execution.md)** - Learn how to train
3. **[06_variables_reference.md](06_variables_reference.md)** - Reference for tuning

### For Developers
1. **[02_integrated_dt_gcn.md](02_integrated_dt_gcn.md)** - Core module internals
2. **[03_RP15_module.md](03_RP15_module.md)** - Baseline implementation
3. **[04_RP12_module.md](04_RP12_module.md)** - Dataset generation

### For Researchers
1. **[07_digital_twin_integration.md](07_digital_twin_integration.md)** - Novel contribution
2. **[02_integrated_dt_gcn.md](02_integrated_dt_gcn.md)** - Technical details
3. **[06_variables_reference.md](06_variables_reference.md)** - Hyperparameters

---

## Key Concepts

### 7-Dimensional Node Features

| Index | Feature | Source |
|-------|---------|--------|
| 0 | is_source | Environment |
| 1 | is_destination | Environment |
| 2 | avg_latency | DT Radio Maps |
| 3 | avg_bandwidth | DT Radio Maps |
| 4 | anomaly_score | AnomalyBridge |
| 5 | jam_probability | JammerDetector |
| 6 | neighbor_jam_avg | JammerDetector |

### Enhanced Reward Formula

```
R = R_base × (1 - 0.3 × anomaly) - 0.5 × jammed + 0.3 × resilience
```

### Key Files

| File | Purpose |
|------|---------|
| `train_integrated.py` | Main training script |
| `compare_models.py` | GCN vs GCN+DT comparison |
| `integrated_dt_gcn/` | Core integration module |
| `RP15/50nodes.brite` | Network topology |
| `RP12_paper/datasets/` | DT datasets |

---

## Quick Start

```bash
# Train with Digital Twin integration
python train_integrated.py --episodes 500 --save --plot

# Compare with baseline
python compare_models.py --episodes 300 --plot
```

---

## Generated Files

After training:
```
integrated_dt_gcn/results/
├── episode_metrics_{timestamp}.csv
├── step_metrics_{timestamp}.csv
├── training_summary.csv
└── training_metrics.png
```

---

## Last Updated

This documentation was generated based on comprehensive analysis of all source files in the repository.
