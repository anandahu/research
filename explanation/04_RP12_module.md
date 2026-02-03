# RP12 Paper Module Deep Dive

## Module Overview

The `RP12_paper/` module implements **Digital Twin-based anomaly detection** for wireless networks. It provides:
1. **Dataset Generation**: Path loss maps, radio maps, and measurement datasets
2. **Anomaly Detection**: Algorithms to detect jammers from measurement differences
3. **Utility Functions**: Radio map manipulation and visualization

This module is the **data source** for the integrated DT-GCN system.

---

## Folder Structure

```
RP12_paper/
├── datasets/                    # Pre-generated datasets (large .pkl files)
│   ├── fspl_PLdataset{N}.pkl   # PathLossMapCollection
│   ├── fspl_RMdataset{N}.pkl   # List of RadioMap objects
│   ├── fspl_measurements{N}.pkl # MeasurementCollection
│   └── *.txt                    # Dataset description files
├── src/
│   ├── __init__.py
│   ├── dataset_generation/      # Scripts to generate datasets
│   │   ├── conf/                # Hydra configuration
│   │   ├── pathloss_map_generation.py
│   │   ├── radio_map_generation.py
│   │   └── measurement_generation.py
│   ├── anomaly_detection/       # Detection algorithms
│   │   ├── conf/
│   │   ├── fspl_anomaly_detection.py
│   │   ├── supervised_detection.py
│   │   └── unsupervised_detection.py
│   └── utils/                   # Utility classes and functions
│       ├── pl_utils.py          # PathLossMap classes
│       ├── radiomap_utils.py    # RadioMap classes
│       └── description_file_utils.py
├── figures/                     # Output visualizations
├── notebooks/                   # Jupyter notebooks
└── README.md
```

---

## Dataset Structure

### Dataset Naming Convention

```
fspl_[type]dataset[N].pkl

Where:
  type: PL (PathLoss), RM (RadioMap), measurements
  N: Dataset number (0-5), corresponds to noise level
```

### Dataset Numbers and Noise Levels

| Dataset Nr | Noise Std (σ) | Description |
|-----------|---------------|-------------|
| 0 | 0 dB | No shadowing noise |
| 1 | 1 dB | Low noise |
| 2 | 2 dB | Medium-low noise |
| 3 | 3 dB | Medium noise |
| 4 | 4 dB | Medium-high noise |
| 5 | 5 dB | High noise |

### Dataset Contents

| File Pattern | Type | Count | Description |
|--------------|------|-------|-------------|
| `fspl_PLdataset{N}.pkl` | `PathLossMapCollection` | ~1681 maps | Path loss maps for each TX position |
| `fspl_RMdataset{N}.pkl` | `List[RadioMap]` | 30,000 | Combined radio maps with TX+jammers |
| `fspl_measurements{N}.pkl` | `MeasurementCollection` | 30,000 | Measurement diffs at 25 points |

### File Sizes
- PathLoss datasets: ~6.7 MB each
- RadioMap datasets: ~413 MB each
- Measurement datasets: ~16-33 MB each

---

## File-by-File Analysis

---

## 1. `src/utils/pl_utils.py`

### Purpose
Path loss map generation and manipulation utilities.

### Class: `PathLossMap`

**Purpose**: Store a single path loss map for one transmitter position.

#### Constructor
```python
def __init__(self, tx_pos, pathloss)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `tx_pos` | `tuple` | Transmitter position (x, y) in meters |
| `pathloss` | `np.ndarray` | Path loss map (41×41) in dB |

#### Method: `show_pathloss_map(show_tx_pos=True)`
Visualize path loss map as heatmap.

### Class: `PathLossMapCollection`

**Purpose**: Collection of path loss maps for a dataset.

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Dataset configuration |
| `pathlossmaps` | `list` | List of PathLossMap objects |

#### Config Keys
```python
config = {
    'scene_size': [41, 41],    # in meters
    'resolution': 1,            # 1 meter per pixel
    'f_c': ...,                 # carrier frequency (Hz)
    'noise_std': ...,           # shadowing std (dB)
    'd_corr': 1,                # correlation distance (m)
}
```

### Function: `generate_fspl_map(scene_size, resolution, tx_pos, f_c, ple=2.0)`

**Purpose**: Generate Free Space Path Loss map.

**Formula** (FSPL):
```
PL = 10 × ple × log10(d) + 20 × log10(f_c) - 147.55

Where:
  PL = path loss in dB
  ple = path loss exponent (default 2.0 for free space)
  d = distance from transmitter in meters
  f_c = carrier frequency in Hz
```

**Implementation**:
```python
def generate_fspl_map(scene_size, resolution, tx_pos, f_c, ple=2.0):
    # Create meshgrid
    x = np.arange(0, scene_size[0], resolution)
    y = np.arange(0, scene_size[1], resolution)
    x, y = np.meshgrid(x, y, indexing='ij')
    
    # Calculate distance from transmitter
    d = np.sqrt((x - tx_pos[0])**2 + (y - tx_pos[1])**2)
    d[d == 0] = 1  # Avoid log(0)
    
    # Calculate path loss
    path_loss = ple*10*np.log10(d) + 20*np.log10(f_c) - 147.55
    
    return path_loss
```

### Function: `create_cov_matrix_corr_shadow(scene_size, dcorr, std)`

**Purpose**: Create covariance matrix for correlated shadowing.

**Formula**:
```
Σ(i,j) = σ² × exp(-d_ij / d_corr)

Where:
  σ = shadowing standard deviation
  d_ij = distance between points i and j
  d_corr = correlation distance
```

### Function: `create_fspl_sample(config, tx_pos=None, cov=None)`

**Purpose**: Generate one path loss map sample with optional shadowing.

**Algorithm**:
```
1. Generate random TX position (or use provided)
2. Generate FSPL map
3. If cov provided: add correlated shadowing noise
4. Return PathLossMap object
```

---

## 2. `src/utils/radiomap_utils.py`

### Purpose
Radio map classes and measurement utilities.

### Class: `RadioMap`

**Purpose**: Combined radio map from multiple transmitters.

#### Constructor
```python
def __init__(self, shape, resolution)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `transmitters` | `list` | Regular Transmitter objects |
| `jammers` | `list` | Jammer Transmitter objects |
| `radio_map` | `np.ndarray` | Combined RSS map (41×41) in dBm |
| `resolution` | `float` | Grid resolution (1 meter) |

#### Method: `add_transmitter(tx_type, tx_pos, tx_power, pathloss_map)`

**Purpose**: Add transmitter and update combined radio map.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tx_type` | `str` | 'tx' or 'jammer' |
| `tx_pos` | `tuple` | Position (x, y) |
| `tx_power` | `float` | Transmit power (dBm) |
| `pathloss_map` | `np.ndarray` | Path loss map (dB) |

**Algorithm**:
```python
# Single transmitter RSS
single_radio_map = tx_power - pathloss_map

# Combine with existing (power summation in linear scale)
if first_transmitter:
    self.radio_map = single_radio_map
else:
    # P_total = 10*log10(10^(P1/10) + 10^(P2/10))
    self.radio_map = 10*np.log10(10**(self.radio_map/10) + 
                                 10**(single_radio_map/10))
```

### Class: `Transmitter`

**Purpose**: Store transmitter properties.

| Attribute | Type | Description |
|-----------|------|-------------|
| `tx_type` | `str` | 'tx' or 'jammer' |
| `tx_pos` | `tuple` | Position in meters |
| `tx_power` | `float` | Transmit power in dBm |

### Class: `MeasurementCollection`

**Purpose**: Store measurement data for all scenarios.

#### Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | 'grid' or 'random' |
| `meas_x` | `list` | X coordinates of measurement points |
| `meas_y` | `list` | Y coordinates of measurement points |
| `grid_size` | `int` | Grid spacing (if grid method) |
| `measurements_diff_list` | `list` | List of (25,) arrays |
| `transmitters_list` | `list` | List of TX lists per scenario |
| `jammers_list` | `list` | List of jammer lists per scenario |

#### Method: `add_measurement(transmitters, jammers, measurements_orig, measurements_dt)`

**Purpose**: Add one scenario's measurement data.

**Stored Value**:
```python
diff = measurements_orig - measurements_dt
# Positive diff = actual > expected = possible jammer!
```

### Function: `generate_measurement_points(method, shape, grid_size)`

**Purpose**: Generate measurement point coordinates.

**Grid Method**:
```python
# 41×41 map with grid_size=8:
# Points at: (4, 4), (4, 12), (4, 20), (4, 28), (4, 36),
#            (12, 4), ..., (36, 36)
# Total: 5×5 = 25 points
```

### Function: `do_measurements(radiomap, meas_x, meas_y)`

**Purpose**: Sample RSS values at measurement points.

**Output**: `np.ndarray (25,)` - RSS values at each point

---

## 3. `src/dataset_generation/pathloss_map_generation.py`

### Purpose
Generate path loss map dataset for all transmitter positions.

### Configuration (Hydra)
```yaml
f_c: 2.4e9        # Carrier frequency: 2.4 GHz
dataset_nr: 0     # Output dataset number
nr_samples: 1681  # Number of samples (41×41 positions)
fspl:
  noise_std: 0    # Shadowing noise std
```

### Algorithm
```
1. Initialize PathLossMapCollection with config
2. Create covariance matrix for correlated shadowing
3. For each sample (1681 iterations):
   a. Generate random TX position
   b. Generate FSPL map
   c. Add correlated shadowing noise
   d. Append to collection
4. Save as pickle file
```

### Output
```
fspl_PLdataset{N}.pkl - PathLossMapCollection
fspl_PLdataset{N}.txt - Configuration description
```

---

## 4. `src/dataset_generation/radio_map_generation.py`

### Purpose
Generate radio maps by combining path loss maps with transmitters and jammers.

### Configuration
```python
tx_power = 20              # Transmit power (dBm)
range_num_tx = [10, 10]     # Min/max transmitters (always 10)
range_jam_power = [20, 20]  # Jammer power (dBm)
range_num_jam = [0, 1]      # 0 or 1 jammer per scenario
num_radiomaps = 30000       # Total scenarios
```

### Algorithm
```
1. Load path loss dataset
2. For each scenario (30,000):
   a. Create empty RadioMap
   b. Choose number of transmitters (10)
   c. For each transmitter:
      - Select random path loss map (unique)
      - Add transmitter to radio map
   d. Choose number of jammers (0 or 1)
   e. For each jammer:
      - Select random path loss map (unique)
      - Add jammer to radio map
3. Save as pickle file
```

### Key Insight
- ~50% of scenarios have 1 jammer
- ~50% have no jammer
- Each scenario has exactly 10 regular transmitters

---

## 5. `src/dataset_generation/measurement_generation.py`

### Purpose
Generate measurement dataset by comparing DT predictions with actual radio maps.

### Configuration
```yaml
measurement_method: 'grid'
grid_size: 8              # 8m spacing → 25 points
tx_pos_inaccuracy_std: 0  # TX position uncertainty
rm_dataset_nr: 0          # Input RadioMap dataset
meas_dataset_nr: 0        # Output measurement dataset
```

### Algorithm
```
1. Load path loss and radio map datasets
2. Generate measurement point grid (25 points)
3. For each radio map:
   a. Create Digital Twin prediction:
      - For each transmitter:
        - Estimate TX position (add noise if configured)
        - Regenerate FSPL map (no shadowing)
        - Add to DT radio map
   b. Sample original radio map at measurement points
   c. Sample DT radio map at measurement points
   d. Store diff = original - DT
4. Save MeasurementCollection
```

### Key Insight
The Digital Twin does NOT know about:
- Jammers (not part of DT model)
- Shadowing noise (cannot replicate)
- TX position errors (if configured)

Therefore, **diffs are larger when jammers are present**!

---

## 6. `src/anomaly_detection/` (Overview)

### Files and Purpose

| File | Purpose |
|------|---------|
| `fspl_anomaly_detection.py` | Simple threshold-based detection |
| `unsupervised_detection.py` | CNN Autoencoder + clustering |
| `supervised_detection.py` | Labeled training approach |

### Detection Methods (from unsupervised_detection.py)

1. **CNN Autoencoder**
   - Train on normal samples
   - Reconstruction error → anomaly score
   
2. **Density-based**
   - Fit distribution on normal diffs
   - Outlier score based on likelihood

3. **DBSCAN Clustering**
   - Cluster measurement diffs
   - Outlier cluster = anomaly

### Key Concept
The integrated module (`AnomalyBridge`) uses a simplified version:
- 2-sigma threshold on measurement diffs
- No deep learning, just statistical threshold

---

## 7. `src/utils/description_file_utils.py`

### Purpose
Parse dataset description text files.

### Content
```python
def get_config_from_file(filepath):
    """Parse .txt file with key: value lines."""
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                config[key.strip()] = parse_value(value.strip())
    return config
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATASET GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Path Loss Generation                                                │
│     ┌─────────────┐                                                    │
│     │  FSPL + σ   │ → fspl_PLdataset{N}.pkl (1681 maps)               │
│     └─────────────┘                                                    │
│                                                                         │
│  2. Radio Map Generation                                                │
│     ┌─────────────┐   ┌──────────┐   ┌──────────┐                     │
│     │ PL dataset  │ + │ 10 TXs   │ + │ 0-1 Jam  │                     │
│     └─────────────┘   └──────────┘   └──────────┘                     │
│              ↓                                                          │
│     fspl_RMdataset{N}.pkl (30,000 RadioMaps)                           │
│                                                                         │
│  3. Measurement Generation                                              │
│     ┌─────────────┐   ┌──────────────┐                                 │
│     │ RM dataset  │ - │ DT prediction │ → Diff at 25 points           │
│     │ (with jam)  │   │ (no jam)      │                                │
│     └─────────────┘   └──────────────┘                                 │
│              ↓                                                          │
│     fspl_measurements{N}.pkl (30,000 diff arrays)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration with DT-GCN

The integrated module uses RP12_paper data as follows:

| RP12 Component | Used By | Purpose |
|----------------|---------|---------|
| `RadioMap.radio_map` | `MeshDigitalTwin` | Update link weights |
| `RadioMap.jammers` | `JammerDetector` | Ground truth jammed nodes |
| `MeasurementCollection.measurements_diff_list` | `AnomalyBridge` | Anomaly score computation |
| Normal diffs | `AnomalyBridge.train()` | Learn detection threshold |

---

## Mathematical Foundations

### Free Space Path Loss (FSPL)

```
PL(d) = 10×n×log10(d) + 20×log10(f) - 147.55 [dB]

Where:
  n = path loss exponent (2 for free space)
  d = distance in meters
  f = frequency in Hz
```

### Signal Power Combination

When multiple transmitters contribute:
```
P_total = 10×log10(Σ 10^(Pi/10)) [dBm]
```

### Correlated Shadowing

```
X ~ N(0, Σ)
Σ(i,j) = σ² × exp(-d_ij / d_corr)

Where:
  σ = shadowing std (0-5 dB in datasets)
  d_corr = correlation distance (1 m)
```

### Measurement Difference

```
Δ = P_actual - P_DT

For jammed scenarios:
  P_actual = P_TXs + P_jammer + noise
  P_DT = P_TXs (no jammer, no noise)
  Δ > threshold → Jammer detected!
```
