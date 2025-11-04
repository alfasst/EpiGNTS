# config.py

# --- Run Control Parameters ---
N_TRAINING_RUNS = 200
N_TESTING_RUNS = 5

# --- Simulation Parameters ---
SIMULATION_DAYS = 150

# --- SBM Network Configurations ---
TEST_NETWORKS = [
    {"name": "Fragmented_1k", "block_sizes": [100]*10, "p_in": 0.01, "p_out": 0.005},
    {"name": "Core-Periphery_2k", "block_sizes": [1000, 200, 200, 200, 200, 200], "p_in": 0.01, "p_out": 0.005},
    {"name": "Bimodal_3k", "block_sizes": [1500, 1500], "p_in": 0.01, "p_out": 0.005},
    {"name": "Pyramidal_4k", "block_sizes": [1200, 800, 800, 240, 240, 240, 240, 240], "p_in": 0.01, "p_out": 0.005},
    {"name": "Heterogeneous_5k", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.005},
    {"name": "SBM_5k_Zero_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.01},
    {"name": "SBM_5k_Low_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.0075},
    {"name": "SBM_5k_Medium_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.005},
    {"name": "SBM_5k_High_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.001},
    {"name": "SBM_5k_Max_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.0}
]

# --- SBM Network Short Names (for file naming) ---
NETWORK_SHORT_NAMES = {
    "Fragmented_1k": "SBM-1k", "Core-Periphery_2k": "SBM-2k", "Bimodal_3k": "SBM-3k",
    "Pyramidal_4k": "SBM-4k", "Heterogeneous_5k": "SBM-5k-Medium",
    "SBM_5k_Zero_Modularity": "SBM-5k-Zero", "SBM_5k_Low_Modularity": "SBM-5k-Low",
    "SBM_5k_Medium_Modularity": "SBM-5k-Medium", "SBM_5k_High_Modularity": "SBM-5k-High",
    "SBM_5k_Max_Modularity": "SBM-5k-Max",
}

# --- SNAP Network Configurations ---
MAX_NODES_SNAP = 10000
TOP_N_COMMUNITIES_SNAP = 20

SNAP_NETWORKS = [
    {
        "name": "Orkut", "short_name": "Orkut-Sampled",
        "path": "snap_data/com-orkut.ungraph.txt",
        "communities_path": "snap_data/com-orkut.all.cmty.txt"
    },
    {
        "name": "LiveJournal", "short_name": "LiveJournal-Sampled",
        "path": "snap_data/com-lj.ungraph.txt",
        "communities_path": "snap_data/com-lj.all.cmty.txt"
    },
    {
        "name": "Youtube", "short_name": "Youtube-Sampled",
        "path": "snap_data/com-youtube.ungraph.txt",
        "communities_path": "snap_data/com-youtube.all.cmty.txt"
    }
]

# Dynamically set parameters
BLOCK_SIZES = []
NUM_BLOCKS = 0

# --- Epidemic Parameters ---
BETA = 0.05
SIGMA = 1/5.0
GAMMA = 1/14.0
INITIAL_INFECTED = 10
ASYMPTOMATIC_PROB = 0.4
LONG_RANGE_INFECTION_PROB = 0.01
WANING_IMMUNITY_PROB = 1/180

# --- Heterogeneity Parameters ---
HUB_BLOCK_ID = 1
HUB_BETA_MULTIPLIER = 2.0

# --- Intervention Parameters ---
TESTING_START_DAY = 20
KITS_SCHEDULE = [(20, 10), (40, 25), (60, 40), (80, 50)]

# --- Agent Parameters ---
LOCAL_GNN_OUTPUT_DIM = 8
LOCAL_GNTS_CONTEXT_DIM = LOCAL_GNN_OUTPUT_DIM + 1
WEIGHT_DECAY = 1e-4

