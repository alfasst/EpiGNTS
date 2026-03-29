# main.py
# Unified experiment runner (per-network–per-strategy daily CSVs)
# --------------------------------------------------
# Changes vs previous version:
# 1. GNTS training uses N-agent ensemble weight averaging instead of
#    keeping only the last sequential agent. Each of the N_TRAINING_RUNS
#    produces an independent agent; their weights are averaged into a
#    stable master model before testing begins.
# 2. average_gnts_agents() covers both LocalGNTS (ModuleList weights)
#    and GlobalGNTS (single-module weights) uniformly.
# 3. "Gamma" removed from STRATEGIES list.

import os
import pickle
import argparse
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import torch
import pandas as pd

import config
from simulation import run_simulation
from gnts import LocalGNTS, GlobalGNTS

# --------------------------------------------------
# Paths
# --------------------------------------------------
GPICKLE_DIR = os.path.join("results", "gpickle")
MODEL_DIR   = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results", "experiments")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# Strategies
# --------------------------------------------------
STRATEGIES = [
    "Uniform",
    "Random",
    "Proportional",
    "Beta",
    "LocalGNTS",
    "GlobalGNTS",
]

SNAP_NAMES = {"Orkut", "LiveJournal", "Youtube"}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_graph(short_name: str):
    path = os.path.join(GPICKLE_DIR, f"{short_name}.gpickle")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Network pickle not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_model_path(strategy: str, network: str) -> str:
    return os.path.join(MODEL_DIR, f"{strategy}_{network}.pt")


def _infer_num_blocks(G) -> int:
    block_ids = [d.get('block_id') for _, d in G.nodes(data=True)
                 if isinstance(d.get('block_id'), int)]
    return max(block_ids) + 1 if block_ids else 0


def _make_agent(strategy: str, G, num_blocks: int):
    """Instantiate a fresh LocalGNTS or GlobalGNTS for the given network."""
    if strategy == "LocalGNTS":
        return LocalGNTS(
            G, num_blocks,
            config.GNN_OUTPUT_DIM,
            config.LOCAL_AGENT_CONTEXT_DIM,
            config.WEIGHT_DECAY,
        )
    return GlobalGNTS(
        G, num_blocks,
        config.GNN_OUTPUT_DIM,
        config.GLOBAL_AGENT_CONTEXT_DIM,
        config.WEIGHT_DECAY,
    )


def average_gnts_agents(agents: list, strategy: str, G, num_blocks: int):
    """Average the weights of N independently trained agents into one master.

    LocalGNTS stores per-block ModuleLists (local_gnns, prior_boosters).
    GlobalGNTS stores single modules (global_gnn, prior_booster).
    Both cases are handled uniformly: collect state_dicts, stack tensors,
    take the element-wise mean, load into a fresh template.
    """
    master = _make_agent(strategy, G, num_blocks)

    if strategy == "LocalGNTS":
        # Each attribute is a ModuleList — average per sub-module
        for attr in ('local_gnns', 'prior_boosters'):
            module_list = getattr(master, attr)
            for idx in range(len(module_list)):
                avg_sd = OrderedDict()
                for key in module_list[idx].state_dict().keys():
                    avg_sd[key] = torch.stack(
                        [getattr(a, attr)[idx].state_dict()[key] for a in agents]
                    ).mean(dim=0)
                module_list[idx].load_state_dict(avg_sd)

    else:  # GlobalGNTS
        # Each attribute is a plain nn.Module — average directly
        for attr in ('global_gnn', 'prior_booster'):
            ref_module = getattr(master, attr)
            avg_sd = OrderedDict()
            for key in ref_module.state_dict().keys():
                avg_sd[key] = torch.stack(
                    [getattr(a, attr).state_dict()[key] for a in agents]
                ).mean(dim=0)
            ref_module.load_state_dict(avg_sd)

    return master


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Run experiments for a single network (short name)",
    )
    args = parser.parse_args()

    sbm_names  = [net['name'] for net in config.TEST_NETWORKS]
    snap_names = [net['name'] for net in config.SNAP_NETWORKS]
    all_networks = sbm_names + snap_names

    if args.network:
        if args.network not in all_networks:
            raise ValueError(f"Unknown network: {args.network}")
        networks = [args.network]
    else:
        networks = all_networks

    for net_name in networks:
        print(f"\n=== Network: {net_name} ===")
        G = load_graph(net_name)
        num_blocks = _infer_num_blocks(G)

        # Budget scaling for large SNAP networks
        if net_name in SNAP_NAMES:
            kits_schedule = [(d, 2 * k) for d, k in config.KITS_SCHEDULE]
        else:
            kits_schedule = config.KITS_SCHEDULE

        for strategy in STRATEGIES:
            print(f"  Strategy: {strategy}")

            out_csv = os.path.join(RESULTS_DIR, f"{net_name}__{strategy}_daily.csv")

            if os.path.exists(out_csv):
                print(f"    ✓ Daily CSV already exists, skipping")
                continue

            pretrained = None
            model_path = get_model_path(strategy, net_name)

            # ----------------------
            # GNTS training
            # ----------------------
            if strategy in ("LocalGNTS", "GlobalGNTS"):
                if os.path.exists(model_path):
                    # Load previously saved master model
                    print("    ✓ Using pretrained GNTS")
                    pretrained = _make_agent(strategy, G, num_blocks)
                    pretrained.load_model(model_path)

                else:
                    # Train N independent agents, then average their weights
                    print(f"    → Training GNTS ensemble ({config.N_TRAINING_RUNS} agents)")
                    trained_agents = []

                    for _ in tqdm(range(config.N_TRAINING_RUNS)):
                        _, agent, _, _ = run_simulation(
                            strategy,
                            G,
                            kits_schedule=kits_schedule,
                        )
                        if agent is not None:
                            trained_agents.append(agent)

                    if trained_agents:
                        print(f"    → Averaging {len(trained_agents)} agents into master")
                        pretrained = average_gnts_agents(
                            trained_agents, strategy, G, num_blocks
                        )
                        pretrained.save_model(model_path)
                        print(f"    ✓ Master model saved to {model_path}")
                    else:
                        print("    ✗ No agents produced — skipping GNTS testing")
                        continue

            # ----------------------
            # Testing runs (aggregate per day)
            # ----------------------
            daily_accumulator = defaultdict(list)

            for run in range(config.N_TESTING_RUNS):
                daily_records, _, metrics, _ = run_simulation(
                    strategy,
                    G,
                    pretrained_gnts=pretrained,
                    kits_schedule=kits_schedule,
                )

                for rec in daily_records:
                    day = rec['Day']
                    daily_accumulator[day].append({
                        'S': rec['S'],
                        'E': rec['E'],
                        'I': rec['I'],
                        'A': rec['A'],
                        'Q': rec['Q'],
                        'R': rec['R'],
                        'total_tests_administered': metrics['total_tests_administered'],
                        'total_positive_tests':     metrics['total_positive_tests'],
                        'total_wasted_tests':        metrics['total_wasted_tests'],
                    })

            # ----------------------
            # Average over runs (per day)
            # ----------------------
            rows = []
            for day in sorted(daily_accumulator.keys()):
                df_day = pd.DataFrame(daily_accumulator[day])
                row = {'Day': day}
                for col in df_day.columns:
                    row[col] = df_day[col].mean()
                rows.append(row)

            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_csv, index=False)
            print(f"    ✓ Saved {out_csv}")


if __name__ == '__main__':
    main()