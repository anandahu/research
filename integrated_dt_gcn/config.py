"""Configuration parameters for DT + GCN integration."""

import torch

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  CUDA not available. Using CPU! (Install torch with CUDA support)")

# Paths
BRITE_FILE = "../RP15/50nodes.brite"
DT_DATASET_DIR = "../RP12_paper/datasets"

# Network parameters
NUM_NODES = 50
NUM_EDGES = 100
DT_MAP_SIZE = 41  # Radio map is 41x41
BRITE_COORD_RANGE = 100  # BRITE coords are 0-99

# Position scaling: BRITE (0-99) -> DT (0-40)
COORD_SCALE = (DT_MAP_SIZE - 1) / BRITE_COORD_RANGE  # 0.4

# Node feature dimensions
NODE_FEATURE_DIM = 7  # 4 original + 3 DT-based

# RSS thresholds (dBm)
RSS_CONNECTIVITY_THRESHOLD = -85  # Min RSS for edge to exist
RSS_REFERENCE = -70  # Reference for normalization

# Reward parameters
LAMBDA_JAM = 0.5  # Jamming penalty coefficient
ALPHA_ANOMALY = 0.3  # Anomaly dampening factor
RESILIENCE_BONUS = 0.3  # Bonus for avoiding jammed nodes

# Training parameters
EPS_DECAY = 1000
EPS_START = 0.95
EPS_END = 0.001
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 14000
LEARNING_RATE = 0.001
