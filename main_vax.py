# main_vax.py
# ---------------------------------------------------------------------------
# Vaccine-workflow experiment entry point.
# Runs training and testing for all vaccine strategies on the three
# SNAP networks (Orkut, LiveJournal, Youtube) only.
#
# Mirrors the structure of main.py with the following changes:
#   1. Uses simulation_vax.run_simulation_vax instead of run_simulation
#   2. Uses LocalGNTSVax / GlobalGNTSVax from gnts_vax
#   3. CSV export includes V compartment column
#   4. Vaccine-specific metrics exported to a separate summary CSV
#   5. SBM networks are not run here — use main.py for test-kit experiments
# ---------------------------------------------------------------------------

import csv
import collections
import os

import torch
from tqdm import tqdm
import networkx as nx

import config
from netepi_vax import build_sim_graph
from simulation_vax import run_simulation_vax
from gnts_vax import LocalGNTSVax, GlobalGNTSVax, average_gnts_vax_bandits

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
GPICKLE_DIR = "gpickle"
CSV_DIR     = "csvs_vax"
MODELS_DIR  = "models_vax"

# ---------------------------------------------------------------------------
# SNAP networks only
# ---------------------------------------------------------------------------
VAX_NETWORKS = config.SNAP_NETWORKS   # Orkut, LiveJournal, Youtube

# ---------------------------------------------------------------------------
# Strategies to evaluate
# ---------------------------------------------------------------------------
strategies_to_run = [
    "LocalGNTSVax-14",
    "GlobalGNTSVax-14",
    "BetaBinomialVax-14",
    "GammaPoissonVax-14",
    "Proportional",
    "RiskWeighted",
    "Uniform",
    "Random",
]


# ---------------------------------------------------------------------------
# CSV export — daily compartment averages
# ---------------------------------------------------------------------------

def export_strategy_results_vax(strategy_name, histories_list, filename):
    """
    Write one CSV per (network, strategy) with averaged daily compartment
    counts across all testing runs.

    Columns: Day, S_avg, E_avg, I_avg, A_avg, V_avg, R_avg,
             Infectious_avg, Immune_avg, VaxCoverage_avg

    VaxCoverage_avg = V_avg / (S_avg + E_avg + I_avg + A_avg + V_avg + R_avg)
    Immune_avg      = V_avg + R_avg
    Infectious_avg  = I_avg + A_avg
    """
    header = [
        "Day",
        "S_avg", "E_avg", "I_avg", "A_avg", "V_avg", "R_avg",
        "Infectious_avg", "Immune_avg", "VaxCoverage_avg",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for day in range(config.SIMULATION_DAYS):
            # Aggregate across runs
            day_run_totals = []
            for history in histories_list:
                run_total = collections.Counter()
                for block_data in history[day]:
                    run_total.update(block_data)
                day_run_totals.append(run_total)

            # Average across runs
            avg = collections.Counter()
            for run_total in day_run_totals:
                avg.update(run_total)
            for k in avg:
                avg[k] /= len(histories_list)

            s = avg.get('S', 0)
            e = avg.get('E', 0)
            i = avg.get('I', 0)
            a = avg.get('A', 0)
            v = avg.get('V', 0)
            r = avg.get('R', 0)

            total        = s + e + i + a + v + r
            infectious   = i + a
            immune       = v + r
            vax_coverage = v / total if total > 0 else 0.0

            writer.writerow([
                day, s, e, i, a, v, r,
                infectious, immune, vax_coverage,
            ])


# ---------------------------------------------------------------------------
# Summary metrics export — one row per (network, strategy, run)
# ---------------------------------------------------------------------------

def export_summary_metrics(all_metrics, filename):
    """
    Write a flat CSV with one row per (network, strategy, run) containing
    the scalar metrics returned by run_simulation_vax.

    Columns:
        Network, Strategy, Run,
        TotalDoses, ProductiveDoses, WastedDoses, WasteRate,
        PeakInfections, TimeToPeak,
        IntegratedInfections, HerdImmunityDay
    """
    header = [
        "Network", "Strategy", "Run",
        "TotalDoses", "ProductiveDoses", "WastedDoses", "WasteRate",
        "PeakInfections", "TimeToPeak",
        "IntegratedInfections", "HerdImmunityDay",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net_name, strategy, run_idx, m in all_metrics:
            total  = m['total_doses_administered']
            waste  = m['total_wasted_doses']
            writer.writerow([
                net_name,
                strategy,
                run_idx,
                total,
                m['total_productive_doses'],
                waste,
                round(waste / total, 4) if total > 0 else 0.0,
                m['peak_infections'],
                m['time_to_peak'],
                m['integrated_infections'],
                m['herd_immunity_day'],
            ])


# ---------------------------------------------------------------------------
# GNTS training helper
# ---------------------------------------------------------------------------

def _train_gnts_vax(strategy_name, sim_graph, agent_cls, agent_kwargs):
    """
    Run N_TRAINING_RUNS episodes, return federated-averaged master agent.
    """
    trained_agents = []
    for _ in tqdm(range(config.N_TRAINING_RUNS),
                  desc=f"  Training {strategy_name}"):
        _, agent, _ = run_simulation_vax(strategy_name, sim_graph)
        trained_agents.append(agent)
    master_template = agent_cls(**agent_kwargs)
    return average_gnts_vax_bandits(trained_agents, master_template)


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_gnts_vax_model(agent, path):
    """
    Save neural-network weights of a vaccine GNTS agent.
    For LocalGNTSVax  : saves encoders + prior_boosters state dicts.
    For GlobalGNTSVax : saves encoder  + prior_boosters state dicts.
    """
    if isinstance(agent, LocalGNTSVax):
        payload = {
            'type':           'LocalGNTSVax',
            'encoders':       [m.state_dict() for m in agent.encoders],
            'prior_boosters': [m.state_dict() for m in agent.prior_boosters],
        }
    elif isinstance(agent, GlobalGNTSVax):
        payload = {
            'type':           'GlobalGNTSVax',
            'encoder':        agent.encoder.state_dict(),
            'prior_boosters': [m.state_dict() for m in agent.prior_boosters],
        }
    else:
        raise TypeError(f"save_gnts_vax_model: unknown agent type {type(agent)}")
    torch.save(payload, path)
    print(f"  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Main loop — SNAP networks only
# ---------------------------------------------------------------------------

os.makedirs(CSV_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Collect scalar metrics across all runs for the summary CSV
all_run_metrics = []   # list of (net_name, strategy, run_idx, metrics_dict)

for net_config in VAX_NETWORKS:
    net_name     = net_config["name"]
    gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")

    if not os.path.exists(gpickle_path):
        print(f"⚠️  Skipping {net_name}: gpickle not found at {gpickle_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Network: {net_name}  [vaccine workflow]")
    print(f"{'='*60}")

    G_nx              = nx.read_gpickle(gpickle_path)
    sim_graph         = build_sim_graph(G_nx)
    config.NUM_BLOCKS = sim_graph['num_blocks']

    gnts_kwargs = dict(
        sim_graph    = sim_graph,
        num_blocks   = config.NUM_BLOCKS,
        gnn_out_dim  = config.GNN_OUTPUT_DIM,
        context_dim  = config.GNTS_CONTEXT_DIM,
        weight_decay = config.WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Training — LocalGNTSVax
    # ------------------------------------------------------------------
    print(f"\n--- TRAINING LocalGNTSVax on {net_name} ---")
    local_master = _train_gnts_vax(
        "LocalGNTSVax-14", sim_graph, LocalGNTSVax, gnts_kwargs
    )
    save_gnts_vax_model(
        local_master,
        os.path.join(MODELS_DIR, f"{net_name}_LocalGNTSVax.pt")
    )

    # ------------------------------------------------------------------
    # Training — GlobalGNTSVax
    # ------------------------------------------------------------------
    print(f"\n--- TRAINING GlobalGNTSVax on {net_name} ---")
    global_master = _train_gnts_vax(
        "GlobalGNTSVax-14", sim_graph, GlobalGNTSVax, gnts_kwargs
    )
    save_gnts_vax_model(
        global_master,
        os.path.join(MODELS_DIR, f"{net_name}_GlobalGNTSVax.pt")
    )

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    print(f"\n--- TESTING on {net_name} ---")
    all_test_histories = collections.defaultdict(list)

    for run_idx in tqdm(range(config.N_TESTING_RUNS),
                        desc=f"  Testing {net_name}"):
        for strategy in strategies_to_run:
            if strategy.startswith("LocalGNTSVax"):
                pretrained = local_master
            elif strategy.startswith("GlobalGNTSVax"):
                pretrained = global_master
            else:
                pretrained = None

            history, _, m = run_simulation_vax(
                strategy, sim_graph, pretrained_gnts=pretrained
            )
            all_test_histories[strategy].append(history)
            all_run_metrics.append((net_name, strategy, run_idx, m))

    # ------------------------------------------------------------------
    # Export per-strategy daily CSVs
    # ------------------------------------------------------------------
    for strategy, histories_list in all_test_histories.items():
        safe_strategy = strategy.replace("-", "_")
        csv_path = os.path.join(CSV_DIR, f"{net_name}_{safe_strategy}.csv")
        export_strategy_results_vax(strategy, histories_list, csv_path)
        print(f"  📄 Exported: {csv_path}")

# ------------------------------------------------------------------
# Export summary metrics CSV (all networks, all strategies, all runs)
# ------------------------------------------------------------------
summary_path = os.path.join(CSV_DIR, "summary_metrics_vax.csv")
export_summary_metrics(all_run_metrics, summary_path)
print(f"\n📊 Summary metrics exported: {summary_path}")
