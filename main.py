# main.py
import csv
import collections
import os

import torch
from tqdm import tqdm
import networkx as nx

import config
from network_epidemic import build_sim_graph
from simulation import run_simulation
from gnts import LocalGNTS, GlobalGNTS, average_gnts_bandits

GPICKLE_DIR = "gpickle"
CSV_DIR     = "csvs"
MODELS_DIR  = "models"

ALL_NETWORKS = config.SBM_NETWORKS + config.SNAP_NETWORKS

strategies_to_run = [
    "LocalGNTS-14",
    "GlobalGNTS-14",
    "Beta-Binomial-14",
    "Proportional",
    "Uniform",
    "Random",
]


def export_strategy_results(strategy_name, histories_list, filename):
    """Write one CSV per (network, strategy) with averaged daily compartment counts."""
    header = ["Day", "S_avg", "E_avg", "I_avg", "A_avg", "Q_avg", "R_avg", "Efficiency_avg"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for day in range(config.SIMULATION_DAYS):
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
    """
    Save the neural-network weights of a GNTS agent.
    For LocalGNTS: saves encoders + prior_boosters state dicts.
    For GlobalGNTS: saves encoder + prior_boosters state dicts.
    """
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
        raise TypeError(f"save_gnts_model: unknown agent type {type(agent)}")
    torch.save(payload, path)


def _train_gnts(strategy_name, sim_graph, agent_cls, agent_kwargs):
    """Run N_TRAINING_RUNS episodes, return averaged master agent."""
    trained_agents = []
    for _ in tqdm(range(config.N_TRAINING_RUNS),
                  desc=f"  Training {strategy_name}"):
        _, agent, _ = run_simulation(strategy_name, sim_graph)
        trained_agents.append(agent)
    master_template = agent_cls(**agent_kwargs)
    return average_gnts_bandits(trained_agents, master_template)


# --------------------------------------------------
# Main loop
# --------------------------------------------------

os.makedirs(CSV_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

for net_config in ALL_NETWORKS:
    net_name     = net_config["name"]
    gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")

    if not os.path.exists(gpickle_path):
        print(f"Skipping {net_name}: gpickle not found at {gpickle_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Network: {net_name}")
    print(f"{'='*60}")

    G_nx = nx.read_gpickle(gpickle_path)
    sim_graph         = build_sim_graph(G_nx)
    config.NUM_BLOCKS = sim_graph['num_blocks']

    # Shared kwargs for both agent constructors
    gnts_kwargs = dict(
        sim_graph   = sim_graph,
        num_blocks  = config.NUM_BLOCKS,
        gnn_out_dim = config.GNN_OUTPUT_DIM,
        context_dim = config.GNTS_CONTEXT_DIM,
        weight_decay= config.WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Training phase — LocalGNTS
    # --------------------------------------------------
    print(f"\n--- TRAINING LocalGNTS on {net_name} ---")
    local_master = _train_gnts("LocalGNTS-14", sim_graph, LocalGNTS, gnts_kwargs)

    local_model_path = os.path.join(MODELS_DIR, f"{net_name}_LocalGNTS.pt")
    save_gnts_model(local_master, local_model_path)
    print(f"  Saved: {local_model_path}")

    # --------------------------------------------------
    # Training phase — GlobalGNTS
    # --------------------------------------------------
    print(f"\n--- TRAINING GlobalGNTS on {net_name} ---")
    global_master = _train_gnts("GlobalGNTS-14", sim_graph, GlobalGNTS, gnts_kwargs)

    global_model_path = os.path.join(MODELS_DIR, f"{net_name}_GlobalGNTS.pt")
    save_gnts_model(global_master, global_model_path)
    print(f"  Saved: {global_model_path}")

    # --------------------------------------------------
    # Testing phase
    # --------------------------------------------------
    print(f"\n--- TESTING on {net_name} ---")
    all_test_results = collections.defaultdict(list)

    for _ in tqdm(range(config.N_TESTING_RUNS), desc=f"  Testing {net_name}"):
        for strategy in strategies_to_run:
            if strategy.startswith("LocalGNTS"):
                pretrained = local_master
            elif strategy.startswith("GlobalGNTS"):
                pretrained = global_master
            else:
                pretrained = None

            history, _, _ = run_simulation(
                strategy, sim_graph, pretrained_gnts=pretrained
            )
            all_test_results[strategy].append(history)

    # --------------------------------------------------
    # Export one CSV per strategy
    # --------------------------------------------------
    for strategy, histories_list in all_test_results.items():
        safe_strategy = strategy.replace("-", "_")
        csv_filename  = os.path.join(CSV_DIR, f"{net_name}_{safe_strategy}.csv")
        export_strategy_results(strategy, histories_list, csv_filename)
        print(f"  Exported: {csv_filename}")