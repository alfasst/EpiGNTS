# config.py
# Static experiment configuration (SHORT NAMES ONLY)
# --------------------------------------------------
# Design rules:
# - Short names are the ONLY network identifiers
# - No runtime mutation
# - No name mapping / aliases

# --------------------------------------------------
# Run control
# --------------------------------------------------
N_TRAINING_RUNS = 150
N_TESTING_RUNS = 5
SIMULATION_DAYS = 150

# --------------------------------------------------
# SBM network configurations (9 networks)
# --------------------------------------------------
# Naming convention:
# SBM-1k, SBM-2k, SBM-3k, SBM-4k,
# SBM-5k-Zero, SBM-5k-Low, SBM-5k-Med, SBM-5k-High, SBM-5k-Max

TEST_NETWORKS = [
    {
        "name": "SBM-1k",
        "block_sizes": [100] * 10,
        "p_in": 0.01,
        "p_out": 0.005
    },
    {
        "name": "SBM-2k",
        "block_sizes": [1000, 200, 200, 200, 200, 200],
        "p_in": 0.01,
        "p_out": 0.005
    },
    {
        "name": "SBM-3k",
        "block_sizes": [1500, 1500],
        "p_in": 0.01,
        "p_out": 0.005
    },
    {
        "name": "SBM-4k",
        "block_sizes": [1200, 800, 800, 240, 240, 240, 240, 240],
        "p_in": 0.01,
        "p_out": 0.005
    },
    {
        "name": "SBM-5k-Zero",
        "block_sizes": [500, 1500, 1000, 800, 1200],
        "p_in": 0.01,
        "p_out": 0.01
    },
    {
        "name": "SBM-5k-Low",
        "block_sizes": [500, 1500, 1000, 800, 1200],
        "p_in": 0.01,
        "p_out": 0.0075
    },
    {
        "name": "SBM-5k-Med",
        "block_sizes": [500, 1500, 1000, 800, 1200],
        "p_in": 0.01,
        "p_out": 0.005
    },
    {
        "name": "SBM-5k-High",
        "block_sizes": [500, 1500, 1000, 800, 1200],
        "p_in": 0.01,
        "p_out": 0.001
    },
    {
        "name": "SBM-5k-Max",
        "block_sizes": [500, 1500, 1000, 800, 1200],
        "p_in": 0.01,
        "p_out": 0.0
    },
]

# --------------------------------------------------
# SNAP network configurations (short names)
# --------------------------------------------------

SNAP_NETWORKS = [
    {
        "name": "Orkut",
        "path": "snap_data/com-orkut.ungraph.txt",
        "communities_path": "snap_data/com-orkut.all.cmty.txt"
    },
    {
        "name": "LiveJournal",
        "path": "snap_data/com-lj.ungraph.txt",
        "communities_path": "snap_data/com-lj.all.cmty.txt"
    },
    {
        "name": "Youtube",
        "path": "snap_data/com-youtube.ungraph.txt",
        "communities_path": "snap_data/com-youtube.all.cmty.txt"
    }
]

# --------------------------------------------------
# Epidemic parameters (SEAIRQ)
# --------------------------------------------------
BETA = 0.05
SIGMA = 1 / 5.0
GAMMA = 1 / 14.0
INITIAL_INFECTED = 10
ASYMPTOMATIC_PROB = 0.4
LONG_RANGE_INFECTION_PROB = 0.01
WANING_IMMUNITY_PROB = 1 / 180

# --------------------------------------------------
# Heterogeneity (SBM-specific, ignored if invalid)
# --------------------------------------------------
HUB_BLOCK_ID = 1
HUB_BETA_MULTIPLIER = 2.0

# --------------------------------------------------
# Testing & intervention
# --------------------------------------------------
TESTING_START_DAY = 20
KITS_SCHEDULE = [
    (20, 10),
    (40, 25),
    (60, 40),
    (80, 50)
]

# --------------------------------------------------
# GNTS / learning parameters
# --------------------------------------------------
GNN_OUTPUT_DIM = 16
LOCAL_AGENT_CONTEXT_DIM = GNN_OUTPUT_DIM + 1
GLOBAL_AGENT_CONTEXT_DIM = GNN_OUTPUT_DIM + 1
WEIGHT_DECAY = 1e-4

# --------------------------------------------------
# Design rule (DO NOT VIOLATE)
# --------------------------------------------------
# This file is STATIC.
# Do not mutate configuration values at runtime.
# Network-specific structure must be inferred from the graph itself.
