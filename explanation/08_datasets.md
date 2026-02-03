# Dataset Documentation

## Overview

This document provides comprehensive documentation for all datasets used in the Digital Twin-Enhanced GNN Routing system. The datasets are located in `RP12_paper/datasets/`.

---

## 1. Dataset Summary

### File Types

| Type | Pattern | Count | Total Size | Description |
|------|---------|-------|------------|-------------|
| Path Loss | `fspl_PLdataset{N}.pkl` | 6 | ~40 MB | Raw path loss maps |
| Radio Maps | `fspl_RMdataset{N}.pkl` | 6 | ~2.4 GB | Combined signal maps |
| Measurements | `fspl_measurements{N}.pkl` | 12+ | ~200 MB | DT vs actual diffs |
| Descriptions | `*.txt` | 24+ | ~3 KB | Configuration metadata |

### Dataset Numbering

| Dataset Nr (N) | Noise Std (σ) | Description |
|----------------|---------------|-------------|
| 0 | 0 dB | Perfect conditions, no shadowing |
| 1 | 1 dB | Very low shadowing noise |
| 2 | 2 dB | Low shadowing noise |
| 3 | 3 dB | Moderate shadowing noise |
| 4 | 4 dB | High shadowing noise |
| 5 | 5 dB | Very high shadowing noise |

---

## 2. Path Loss Datasets (`fspl_PLdataset{N}.pkl`)

### Purpose
Store pre-computed Free Space Path Loss (FSPL) maps for each possible transmitter position in the 41×41 meter scene.

### Structure

```python
# File content type
PathLossMapCollection:
    config: dict          # Dataset configuration
    pathlossmaps: list    # List of PathLossMap objects
```

### Config Dictionary

| Key | Type | Value | Description |
|-----|------|-------|-------------|
| `scene_size` | `list` | `[41, 41]` | Scene dimensions in meters |
| `resolution` | `int` | `1` | Resolution: 1 meter per pixel |
| `f_c` | `float` | `2.4e9` | Carrier frequency (2.4 GHz) |
| `noise_std` | `float` | 0-5 | Shadowing std deviation (dB) |
| `d_corr` | `int` | `1` | Correlation distance (meters) |

### PathLossMap Object

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `tx_pos` | `tuple` | `(2,)` | Transmitter position (x, y) |
| `pathloss` | `np.ndarray` | `(41, 41)` | Path loss values in dB |

### Sample Count
Each dataset contains **1681 path loss maps** (41 × 41 positions).

### Path Loss Formula (FSPL)

```
PL(d) = 10 × n × log₁₀(d) + 20 × log₁₀(f) - 147.55 [dB]

Where:
  n = 2 (free space path loss exponent)
  d = distance in meters
  f = 2.4 × 10⁹ Hz (carrier frequency)
```

### Example Values

| Distance (m) | Path Loss (dB) |
|--------------|----------------|
| 1 | ~40 dB |
| 5 | ~54 dB |
| 10 | ~60 dB |
| 20 | ~66 dB |
| 40 | ~72 dB |

### File Size
Each `fspl_PLdataset{N}.pkl`: **~6.75 MB**

### Loading Example

```python
import pickle

with open("fspl_PLdataset0.pkl", "rb") as f:
    plmc = pickle.load(f)

# Access configuration
print(plmc.config)
# {'scene_size': [41, 41], 'resolution': 1, 'f_c': 2400000000.0, 'noise_std': 0.0, 'd_corr': 1}

# Access first path loss map
plm = plmc.pathlossmaps[0]
print(f"TX Position: {plm.tx_pos}")
print(f"Path loss shape: {plm.pathloss.shape}")  # (41, 41)
print(f"Path loss range: {plm.pathloss.min():.1f} to {plm.pathloss.max():.1f} dB")
```

---

## 3. Radio Map Datasets (`fspl_RMdataset{N}.pkl`)

### Purpose
Store complete radio environments with multiple transmitters and optional jammers. Each scenario represents one possible network state.

### Structure

```python
# File content type
List[RadioMap]:  # 30,000 RadioMap objects
```

### RadioMap Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `radio_map` | `np.ndarray (41, 41)` | Combined RSS in dBm |
| `transmitters` | `List[Transmitter]` | Regular transmitters (10 per scenario) |
| `jammers` | `List[Transmitter]` | Jammers (0 or 1 per scenario) |
| `resolution` | `float` | Grid resolution (1.0) |

### Transmitter Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `tx_type` | `str` | `'tx'` or `'jammer'` |
| `tx_pos` | `tuple` | Position (x, y) in meters |
| `tx_power` | `float` | Transmit power in dBm |

### Scenario Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total scenarios | 30,000 | Per dataset |
| Transmitters per scenario | 10 | Fixed |
| Jammers per scenario | 0 or 1 | 50% probability |
| TX power | 20 dBm | All transmitters |
| Jammer power | 20 dBm | Same as TX |

### RSS Combination Formula

When multiple transmitters contribute to a point:
```
P_total = 10 × log₁₀(Σᵢ 10^(Pᵢ/10)) [dBm]

Where Pᵢ = Pₜₓ - PLᵢ for each transmitter
```

### File Size
Each `fspl_RMdataset{N}.pkl`: **~413 MB**

### Jammed vs Normal Distribution

| Scenario Type | Count | Percentage |
|---------------|-------|------------|
| With jammer | ~15,000 | ~50% |
| Without jammer | ~15,000 | ~50% |

### Loading Example

```python
import pickle

with open("fspl_RMdataset0.pkl", "rb") as f:
    radiomaps = pickle.load(f)

print(f"Total scenarios: {len(radiomaps)}")  # 30000

# Examine first scenario
rm = radiomaps[0]
print(f"Radio map shape: {rm.radio_map.shape}")  # (41, 41)
print(f"RSS range: {rm.radio_map.min():.1f} to {rm.radio_map.max():.1f} dBm")
print(f"Transmitters: {len(rm.transmitters)}")  # 10
print(f"Jammers: {len(rm.jammers)}")  # 0 or 1

# Get jammer info
if rm.jammers:
    jammer = rm.jammers[0]
    print(f"Jammer position: {jammer.tx_pos}")
    print(f"Jammer power: {jammer.tx_power} dBm")
```

### RSS Value Interpretation

| RSS (dBm) | Signal Quality | Routing Impact |
|-----------|----------------|----------------|
| > -50 | Excellent | Low latency, high capacity |
| -50 to -70 | Good | Normal operation |
| -70 to -85 | Fair | Increased latency |
| < -85 | Poor | Connectivity threshold |

---

## 4. Measurement Datasets (`fspl_measurements{N}.pkl`)

### Purpose
Store the **differences** between Digital Twin predictions and actual measurements. These differences are the key to anomaly detection.

### Structure

```python
# File content type
MeasurementCollection:
    method: str                    # 'grid'
    meas_x: list                   # X coordinates of measurement points
    meas_y: list                   # Y coordinates of measurement points
    grid_size: int                 # 8 meters
    measurements_diff_list: list   # List of diff arrays
    transmitters_list: list        # TX info per scenario
    jammers_list: list             # Jammer info per scenario
```

### Measurement Point Grid

```
Grid configuration:
  - Scene: 41 × 41 meters
  - Grid size: 8 meters
  - Offset: 4 meters (centered)

Points: (4, 4), (4, 12), (4, 20), (4, 28), (4, 36),
        (12, 4), (12, 12), ..., (36, 36)
        
Total: 5 × 5 = 25 points per scenario
```

### Difference Calculation

```
For each scenario:
  1. Get actual RSS at 25 points from RadioMap
  2. Recreate DT prediction:
     - Use estimated TX positions (with uncertainty)
     - Regenerate FSPL maps (no shadowing, no jammer)
     - Combine transmitter signals
     - Sample at 25 points
  3. Compute: diff = actual - predicted
```

### What the DT Doesn't Know

| Factor | In Actual | In DT | Effect on Diff |
|--------|-----------|-------|----------------|
| Jammers | Yes | No | Large positive diff near jammer |
| Shadowing | Yes | No | Small random diffs |
| TX position error | Possible | Estimated | Small systematic diffs |

### Measurement Dataset Variants

| Dataset Nr | Noise Std | Grid Size | Notes |
|------------|-----------|-----------|-------|
| 0 | 0 dB | 8m | Clean baseline |
| 1-5 | 1-5 dB | 8m | Increasing noise |
| 10 | varies | 5m | Denser grid |
| 11-12 | varies | 10m | Sparser grid |

### File Sizes

| Dataset | Size |
|---------|------|
| `fspl_measurements0-5.pkl` | ~19-20 MB each |
| `fspl_measurements10.pkl` | ~33 MB (denser grid) |
| `fspl_measurements11-12.pkl` | ~16 MB (sparser grid) |

### Loading Example

```python
import pickle
import numpy as np

with open("fspl_measurements0.pkl", "rb") as f:
    mc = pickle.load(f)

print(f"Measurement method: {mc.method}")  # 'grid'
print(f"Grid size: {mc.grid_size}")  # 8
print(f"Measurement points: {len(mc.meas_x)}")  # 25
print(f"Total scenarios: {len(mc.measurements_diff_list)}")  # 30000

# Get measurement points
print(f"X coords: {mc.meas_x}")  # [4, 4, 4, 4, 4, 12, 12, ...]
print(f"Y coords: {mc.meas_y}")  # [4, 12, 20, 28, 36, 4, ...]

# Examine first scenario
diff = mc.measurements_diff_list[0]
print(f"Diff shape: {diff.shape}")  # (25,)
print(f"Diff range: {diff.min():.2f} to {diff.max():.2f} dB")

# Check if jammed
has_jammer = len(mc.jammers_list[0]) > 0
print(f"Scenario 0 has jammer: {has_jammer}")
```

### Expected Difference Values

| Scenario Type | Mean |diff| | Std |diff| | Max |diff| |
|---------------|----------|---------|----------|
| Normal (no jammer) | 0.3-0.5 dB | 0.2-0.4 dB | 1-2 dB |
| Jammed | 2-10 dB | 2-5 dB | 15-20 dB |

### Anomaly Detection Threshold

```python
# Training phase (on normal samples only):
all_abs_diffs = np.abs(np.concatenate(normal_diffs))
mean = np.mean(all_abs_diffs)  # ~0.4 dB
std = np.std(all_abs_diffs)    # ~0.3 dB
threshold = mean + 2 * std     # ~1.0 dB (2-sigma rule)

# Inference:
# If |diff| > threshold at a point → potential anomaly
```

---

## 5. Description Files (`*.txt`)

### Purpose
Human-readable metadata files accompanying each dataset.

### Path Loss Dataset Description

```
scene_size: [41, 41]
resolution: 1
f_c: 2400000000.0
noise_std: 0.0
d_corr: 1
```

### Measurement Dataset Description

```
scene_size: [41, 41]
resolution: 1
f_c: 2400000000.0
noise_std: 0.0
d_corr: 1

measurement_method: grid
grid_size: 8
tx_pos_inaccuracy_std: 0
```

---

## 6. Dataset Generation Pipeline

### Step 1: Path Loss Generation

```python
# pathloss_map_generation.py

for each position in 41×41 grid:
    1. Set TX position
    2. Generate FSPL map
    3. Add correlated shadowing (σ dB)
    4. Store PathLossMap object
    
Save as fspl_PLdataset{N}.pkl
```

### Step 2: Radio Map Generation

```python
# radio_map_generation.py

for scenario in range(30000):
    1. Create empty RadioMap
    2. Select 10 random TX positions (unique)
    3. For each TX:
       - Get pre-computed path loss map
       - Add to combined radio map (power sum)
    4. With 50% probability:
       - Select 1 jammer position
       - Add jammer signal to radio map
    5. Store RadioMap object
    
Save as fspl_RMdataset{N}.pkl
```

### Step 3: Measurement Generation

```python
# measurement_generation.py

for each RadioMap:
    1. Create DT prediction:
       - For each TX (no jammer):
         - Estimate position (add uncertainty)
         - Generate clean FSPL map
         - Add to DT radio map
    2. Sample actual RadioMap at 25 grid points
    3. Sample DT prediction at 25 grid points
    4. Compute diff = actual - DT
    5. Store diff, TX list, jammer list
    
Save as fspl_measurements{N}.pkl
```

---

## 7. Usage in Training

### Dataset Loading

```python
from integrated_dt_gcn.dataset_loader import DTDatasetLoader

loader = DTDatasetLoader("RP12_paper/datasets")

# Load radio maps for 30,000 scenarios
radio_maps = loader.load_radio_maps(dataset_nr=0)

# Load measurements
measurements = loader.load_measurements(dataset_nr=0)

# Get indices
jammed_indices = loader.get_jammed_indices(0)    # ~15,000
normal_indices = loader.get_normal_indices(0)    # ~15,000
```

### Anomaly Detector Training

```python
# Use only normal (non-jammed) samples
normal_diffs = loader.get_normal_measurements(dataset_nr=0)[:2000]

anomaly_bridge = AnomalyBridge()
anomaly_bridge.train(normal_diffs)

print(f"Threshold: {anomaly_bridge.threshold:.2f} dB")
```

### Environment Scenario Selection

```python
# HybridEnv.reset() logic:
if self.episode_count % 2 == 0:
    idx = random.choice(self.jammed_indices)  # 50% jammed
else:
    idx = random.choice(self.normal_indices)  # 50% normal

# Load scenario data
radio_map = self.loader.get_radio_map_array(idx)
meas_diff = self.loader.get_measurement_diff(idx)
jammer_pos = self.loader.get_jammer_positions(idx)
```

---

## 8. Data Statistics

### Per-Scenario Memory Usage

| Data Type | Size per Scenario |
|-----------|------------------|
| Radio Map (41×41 float64) | ~13.5 KB |
| Measurement Diff (25 float64) | ~0.2 KB |
| Transmitter List (10 objects) | ~1 KB |
| Jammer List (0-1 objects) | ~0.1 KB |

### Total Dataset Sizes

| Component | Size | Notes |
|-----------|------|-------|
| All PL datasets | ~40 MB | 6 datasets × 6.75 MB |
| All RM datasets | ~2.5 GB | 6 datasets × 413 MB |
| All Measurements | ~200 MB | 12+ datasets |
| **Total** | **~2.7 GB** | |

---

## 9. Dataset Quality Checks

### Validation Code

```python
import pickle
import numpy as np

def validate_dataset(dataset_nr):
    # Load all three types
    with open(f"fspl_PLdataset{dataset_nr}.pkl", "rb") as f:
        plmc = pickle.load(f)
    with open(f"fspl_RMdataset{dataset_nr}.pkl", "rb") as f:
        rms = pickle.load(f)
    with open(f"fspl_measurements{dataset_nr}.pkl", "rb") as f:
        mc = pickle.load(f)
    
    # Check counts
    assert len(plmc.pathlossmaps) == 1681, "PL count mismatch"
    assert len(rms) == 30000, "RM count mismatch"
    assert len(mc.measurements_diff_list) == 30000, "Meas count mismatch"
    
    # Check shapes
    for plm in plmc.pathlossmaps[:10]:
        assert plm.pathloss.shape == (41, 41)
    for rm in rms[:10]:
        assert rm.radio_map.shape == (41, 41)
    for diff in mc.measurements_diff_list[:10]:
        assert diff.shape == (25,)
    
    # Check jammer distribution
    jammed = sum(1 for j in mc.jammers_list if len(j) > 0)
    print(f"Dataset {dataset_nr}: {jammed}/30000 jammed ({100*jammed/30000:.1f}%)")
    
    # Check RSS ranges
    all_rss = np.array([rm.radio_map for rm in rms[:100]])
    print(f"RSS range: {all_rss.min():.1f} to {all_rss.max():.1f} dBm")
    
    print(f"Dataset {dataset_nr} validated successfully!")

# Run validation
for n in range(6):
    validate_dataset(n)
```

---

## 10. Extending the Datasets

### Generate New Dataset with Different Parameters

```bash
cd RP12_paper/src/dataset_generation

# Edit config files in conf/
# Then run:
python pathloss_map_generation.py
python radio_map_generation.py
python measurement_generation.py
```

### Configuration Options

**pathloss_map_generation.yaml**:
```yaml
f_c: 2.4e9          # Carrier frequency
dataset_nr: 6       # New dataset number
nr_samples: 1681    # Position samples
fspl:
  noise_std: 6      # New noise level
```

**radio_map_generation.yaml**:
```yaml
dataset_nr: 6
num_radiomaps: 30000
```

**measurement_generation.yaml**:
```yaml
rm_dataset_nr: 6
meas_dataset_nr: 6
measurement_method: grid
grid_size: 8
tx_pos_inaccuracy_std: 0
```
