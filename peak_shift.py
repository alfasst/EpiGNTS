# expt_peak_shift.py
import os
import csv
import collections
import torch
from tqdm import tqdm
import networkx as nx

# Import core simulation modules
import config
from network_epidemic import build_sim_graph
from simulation import run_simulation
from gnts import LocalGNTS, GlobalGNTS, average_gnts_bandits

# --- Experiment Setup ---
GPICKLE_DIR = "gpickle"
CSV_DIR     = "csvs_peak_shift"
MODELS_DIR  = "models_peak_shift"

TARGET_NETWORKS = ["SBM-1k", "SBM-2k", "SBM-3k", "SBM-4k", "SBM-5k-Med"]

strategies_to_run = [
    "LocalGNTS-14",
    "GlobalGNTS-14",
    "Beta-Binomial-14",
    "Proportional",
    "Uniform",
    "Random",
]

SCENARIOS = [
    {
        "name": "Peak_25",
        "BETA": 0.05,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.01,
        "SIMULATION_DAYS": 150,
        "KITS_SCHEDULE": [(20, 10), (40, 25), (60, 40), (80, 50)]
    },
    {
        "name": "Peak_40",
        "BETA": 0.035,
        "INITIAL_INFECTED": 5,
        "LONG_RANGE_INFECTION_PROB": 0.005,
        "SIMULATION_DAYS": 150,
        "KITS_SCHEDULE": [(30, 10), (50, 25), (70, 40), (90, 50)]
    },
    {
        "name": "Peak_60",
        "BETA": 0.025,
        "INITIAL_INFECTED": 3,
        "LONG_RANGE_INFECTION_PROB": 0.002,
        "SIMULATION_DAYS": 200,
        "KITS_SCHEDULE": [(40, 10), (60, 25), (80, 40), (100, 50)]
    },
    {
        "name": "Peak_80",
        "BETA": 0.018,
        "INITIAL_INFECTED": 2,
        "LONG_RANGE_INFECTION_PROB": 0.001,
        "SIMULATION_DAYS": 200,
        "KITS_SCHEDULE": [(50, 10), (80, 25), (110, 40), (140, 50)]
    },
    {
        "name": "Peak_100",
        "BETA": 0.015,
        "INITIAL_INFECTED": 2,
        "LONG_RANGE_INFECTION_PROB": 0.0005,
        "SIMULATION_DAYS": 250,
        "KITS_SCHEDULE": [(60, 10), (90, 25), (120, 40), (150, 50)]
    }
]

# --- Helper Functions (Copied to keep script independent) ---
def export_strategy_results(strategy_name, histories_list, filename, sim_days):
    header = ["Day", "S_avg", "E_avg", "I_avg", "A_avg", "Q_avg", "R_avg", "Efficiency_avg"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for day in range(sim_days):
            day_run_totals = []
            for history in histories_list:
                run_total = collections.Counter()
                for block_data in history[day]:
                    run_total.update(block_data)
                day_run_totals.append(run_total)
            avg = collections.Counter()
            for run_total in day_run_totals:
                avg.update(run_total)
            for k in avg:
                avg[k] /= len(histories_list)
            s, e, i, a, q, r = [avg.get(st, 0) for st in ["S", "E", "I", "A", "Q", "R"]]
            denominator = i + e + a + q
            efficiency  = q / denominator if denominator > 0 else 0
            writer.writerow([day, s, e, i, a, q, r, efficiency])

def save_gnts_model(agent, path):
    if isinstance(agent, LocalGNTS):
        payload = {
            'type':           'LocalGNTS',
            'encoders':       [m.state_dict() for m in agent.encoders],
            'prior_boosters': [m.state_dict() for m in agent.prior_boosters],
        }
    elif isinstance(agent, GlobalGNTS):
        payload = {
            'type':           'GlobalGNTS',
            'encoder':        agent.encoder.state_dict(),
            'prior_boosters': [m.state_dict() for m in agent.prior_boosters],
        }
    else:
        raise TypeError(f"save_gnts_model: unknown agent type")
    torch.save(payload, path)

def _train_gnts(strategy_name, sim_graph, agent_cls, agent_kwargs):
    trained_agents = []
    for _ in tqdm(range(config.N_TRAINING_RUNS), desc=f"  Training {strategy_name}"):
        _, agent, _ = run_simulation(strategy_name, sim_graph)
        trained_agents.append(agent)
    master_template = agent_cls(**agent_kwargs)
    return average_gnts_bandits(trained_agents, master_template)

# --- Main Experiment Loop ---
if __name__ == "__main__":
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    for scenario in SCENARIOS:
        print(f"\n{'#'*70}")
        print(f"### RUNNING SCENARIO: {scenario['name']} ###")
        print(f"{'#'*70}")

        # Override config variables dynamically
        config.BETA = scenario["BETA"]
        config.INITIAL_INFECTED = scenario["INITIAL_INFECTED"]
        config.LONG_RANGE_INFECTION_PROB = scenario["LONG_RANGE_INFECTION_PROB"]
        config.SIMULATION_DAYS = scenario["SIMULATION_DAYS"]
        config.KITS_SCHEDULE = scenario["KITS_SCHEDULE"]

        for net_name in TARGET_NETWORKS:
            gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")
            
            if not os.path.exists(gpickle_path):
                print(f"Skipping {net_name}: gpickle not found at {gpickle_path}")
                continue

            print(f"\n{'-'*60}")
            print(f"Network: {net_name} | Peak Target: {scenario['name']}")
            print(f"{'-'*60}")

            G_nx = nx.read_gpickle(gpickle_path)
            sim_graph = build_sim_graph(G_nx)
            config.NUM_BLOCKS = sim_graph['num_blocks']

            gnts_kwargs = dict(
                sim_graph    = sim_graph,
                num_blocks   = config.NUM_BLOCKS,
                gnn_out_dim  = config.GNN_OUTPUT_DIM,
                context_dim  = config.GNTS_CONTEXT_DIM,
                weight_decay = config.WEIGHT_DECAY,
            )

            # Training
            print(f"\n  [Training LocalGNTS]")
            local_master = _train_gnts("LocalGNTS-14", sim_graph, LocalGNTS, gnts_kwargs)
            local_model_path = os.path.join(MODELS_DIR, f"{net_name}_{scenario['name']}_LocalGNTS.pt")
            save_gnts_model(local_master, local_model_path)

            print(f"\n  [Training GlobalGNTS]")
            global_master = _train_gnts("GlobalGNTS-14", sim_graph, GlobalGNTS, gnts_kwargs)
            global_model_path = os.path.join(MODELS_DIR, f"{net_name}_{scenario['name']}_GlobalGNTS.pt")
            save_gnts_model(global_master, global_model_path)

            # Testing
            print(f"\n  [Testing Strategies]")
            all_test_results = collections.defaultdict(list)

            for _ in tqdm(range(config.N_TESTING_RUNS), desc=f"  Testing {net_name}"):
                for strategy in strategies_to_run:
                    if strategy.startswith("LocalGNTS"):
                        pretrained = local_master
                    elif strategy.startswith("GlobalGNTS"):
                        pretrained = global_master
                    else:
                        pretrained = None

                    history, _, _ = run_simulation(strategy, sim_graph, pretrained_gnts=pretrained)
                    all_test_results[strategy].append(history)

            # Exporting
            for strategy, histories_list in all_test_results.items():
                safe_strategy = strategy.replace("-", "_")
                csv_filename = os.path.join(CSV_DIR, f"{net_name}_{scenario['name']}_{safe_strategy}.csv")
                export_strategy_results(strategy, histories_list, csv_filename, config.SIMULATION_DAYS)