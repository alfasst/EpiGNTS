# network_epidemic.py
# Revised version
# Scope:
# - Epidemic dynamics only (SEAIRQ)
# - NO network generation
# - NO dependence on mutable global config for block counts

import networkx as nx
import random
import config

# --------------------------------------------------
# Epidemic initialization
# --------------------------------------------------

def initialize_epidemic(G, num_initial):
    """Initializes node states for the SEAIRQ model.
    All nodes start as Susceptible ('S'), with num_initial set to Infected ('I').
    """
    if G.number_of_nodes() == 0:
        return

    nx.set_node_attributes(G, 'S', 'state')

    num_initial = min(num_initial, G.number_of_nodes())
    if num_initial <= 0:
        return

    try:
        infected = random.sample(list(G.nodes()), num_initial)
        nx.set_node_attributes(G, {n: 'I' for n in infected}, 'state')
    except ValueError:
        pass


# --------------------------------------------------
# SEAIRQ dynamics (single time step)
# --------------------------------------------------

def run_seiqr_step(
    G,
    beta,
    sigma,
    gamma,
    asymptomatic_prob,
    hub_id,
    hub_multiplier,
    long_range_prob,
    waning_prob
):
    """Advance epidemic state by one day using SEAIRQ dynamics."""

    if G.number_of_nodes() == 0:
        return

    # Infer number of blocks directly from graph
    block_ids = nx.get_node_attributes(G, 'block_id')
    num_blocks = max(block_ids.values()) + 1 if block_ids else 0
    is_hub_valid = 0 <= hub_id < num_blocks

    new_states = {}
    newly_exposed = set()

    susceptible_nodes = [n for n, d in G.nodes(data=True) if d.get('state') == 'S']

    for node, data in G.nodes(data=True):
        state = data.get('state', 'S')
        block_id = data.get('block_id', -1)

        # ----------------------
        # Infectious (I, A)
        # ----------------------
        if state in ('I', 'A'):
            eff_beta = beta * hub_multiplier if (is_hub_valid and block_id == hub_id) else beta

            # Local transmission
            for nbr in G.neighbors(node):
                if G.nodes[nbr].get('state') == 'S':
                    if random.random() < eff_beta:
                        newly_exposed.add(nbr)

            # Long-range infection
            if susceptible_nodes and random.random() < long_range_prob:
                newly_exposed.add(random.choice(susceptible_nodes))

            # Recovery
            if random.random() < gamma:
                new_states[node] = 'R'

        # ----------------------
        # Exposed (E)
        # ----------------------
        elif state == 'E':
            if random.random() < sigma:
                new_states[node] = 'A' if random.random() < asymptomatic_prob else 'I'

        # ----------------------
        # Quarantined (Q)
        # ----------------------
        elif state == 'Q':
            if random.random() < gamma:
                new_states[node] = 'R'

        # ----------------------
        # Recovered (R)
        # ----------------------
        elif state == 'R':
            if random.random() < waning_prob:
                new_states[node] = 'S'

    # Apply newly exposed nodes
    for node in newly_exposed:
        if node not in new_states and G.nodes[node].get('state') == 'S':
            new_states[node] = 'E'

    nx.set_node_attributes(G, new_states, 'state')

