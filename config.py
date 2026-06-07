# config.py

# --- Run Control Parameters ---
# CHANGED: Increased training runs for the new model
N_TRAINING_RUNS = 100
N_TESTING_RUNS = 5

# --- Simulation Parameters ---
SIMULATION_DAYS = 150

# --- Network Parameters ---
SBM_NETWORKS = [
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

# --- Epidemic Parameters ---
BETA = 0.05; SIGMA = 1/5.0; GAMMA = 1/14.0; INITIAL_INFECTED = 10
ASYMPTOMATIC_PROB = 0.4; LONG_RANGE_INFECTION_PROB = 0.01; WANING_IMMUNITY_PROB = 1/180

# --- Heterogeneity Parameters ---
HUB_BLOCK_ID = 1; HUB_BETA_MULTIPLIER = 2.0

# --- Intervention Parameters ---
TESTING_START_DAY = 20
KITS_SCHEDULE = [(20, 10), (40, 25), (60, 40), (80, 50)]
WARM_START_DAYS = 7; EXPLORATION_BUDGET_FRACTION = 0.1

# --- Agent Parameters ---
# NEW: Parameters for the LocalGNTS agent
GNN_OUTPUT_DIM = 8
# Context is Local GNN output + 1 for time
GNTS_CONTEXT_DIM = GNN_OUTPUT_DIM + 1
WEIGHT_DECAY = 1e-4


# --- Vaccine Workflow Epidemic Parameters ---
VAX_BETA                      = 0.02
VAX_SIGMA                     = 1 / 8.0
VAX_INITIAL_INFECTED          = 5
VAX_LONG_RANGE_INFECTION_PROB = 0.001

VACCINATION_START_DAY = 20
DOSES_SCHEDULE = [(20, 10), (40, 20), (60, 30), (80, 40)]
WANING_VACCINE_PROB     = 1 / 180
HERD_IMMUNITY_THRESHOLD = 0.7
VACCINE_EFFECT_LAG = 14

