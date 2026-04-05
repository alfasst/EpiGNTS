# main.py
# --------------------------------------------------
# Training logic: Sequential learning over multiple episodes.
# No weight averaging. Realistic feature encoding enabled.
#
# Change vs previous version:
# - run_simulation() now takes agent= and training_mode= instead of
#   pretrained_gnts=.  training_mode=True during training so the agent's
#   weights accumulate in-place across episodes.  training_mode=False
#   (default) during testing so the master model is never mutated.

import os, pickle, torch, pandas as pd
from tqdm import tqdm
from collections import defaultdict
import config
from simulation import run_simulation
from gnts import LocalGNTS, GlobalGNTS, DEVICE
from network_epidemic import build_sim_structures

GPICKLE_DIR, MODEL_DIR, RESULTS_DIR = "results/gpickle", "results/models", "results/experiments"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

STRATEGIES = ["Uniform", "Random", "Proportional", "Beta-14", "LocalGNTS-14", "GlobalGNTS-14"]

def load_graph(name):
    with open(os.path.join(GPICKLE_DIR, f"{name}.gpickle"), 'rb') as f: return pickle.load(f)

def get_model_path(strat, net): return os.path.join(MODEL_DIR, f"{strat}_{net}.pt")

def net_level(net_name):
    print(f"\n=== Network: {net_name} ===")
    G = load_graph(net_name)
    num_blocks = max([d.get('block_id', 0) for _, d in G.nodes(data=True)]) + 1
    sim_structs = build_sim_structures(G)
    ks = [(d, 2*k) for d, k in config.KITS_SCHEDULE] if net_name in {"Orkut", "LiveJournal", "Youtube"} else config.KITS_SCHEDULE

    for strategy in STRATEGIES:
        out_csv = os.path.join(RESULTS_DIR, f"{net_name}__{strategy}_daily.csv")
        if os.path.exists(out_csv): continue
        
        agent = None
        mpath = get_model_path(strategy, net_name)

        if "GNTS" in strategy:
            if "Local" in strategy:
                agent = LocalGNTS(G, num_blocks, config.GNN_OUTPUT_DIM, config.LOCAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY)
            else:
                agent = GlobalGNTS(G, num_blocks, config.GNN_OUTPUT_DIM, config.GLOBAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY, precomputed=sim_structs)
            
            if os.path.exists(mpath):
                agent.load_model(mpath)
                print(f"  ✓ Loaded {strategy}")
            else:
                print(f"  → Training {strategy} sequentially for {config.N_TRAINING_RUNS} runs")
                for _ in tqdm(range(config.N_TRAINING_RUNS)):
                    # training_mode=True: agent weights accumulate in-place
                    # across episodes — no deepcopy inside run_simulation.
                    run_simulation(strategy, G, sim_structures=sim_structs,
                                   agent=agent, kits_schedule=ks,
                                   training_mode=True)
                agent.save_model(mpath)

        # Testing phase — training_mode defaults to False so the master
        # agent is deepcopied inside run_simulation and never mutated.
        if agent:
            if hasattr(agent, 'local_gnns'):
                agent.local_gnns.eval()
            else:
                agent.global_gnn.eval()
        
        daily_acc = defaultdict(list)
        for _ in range(config.N_TESTING_RUNS):
            recs, _, mets, _ = run_simulation(strategy, G, sim_structures=sim_structs,
                                              agent=agent, kits_schedule=ks)
            for r in recs: daily_acc[r['Day']].append({**r, **mets})

        rows = []
        for d in sorted(daily_acc.keys()):
            dfd = pd.DataFrame(daily_acc[d])
            rows.append({'Day': d, **dfd.mean().to_dict()})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  ✓ Finished {strategy}")

if __name__ == '__main__':
    all_nets = [n['name'] for n in config.TEST_NETWORKS] + [n['name'] for n in config.SNAP_NETWORKS]
    for n in all_nets: net_level(n)