# export_networks_with_layouts.py
# Description: Generates SBM networks, calculates a custom geometric layout for each, 
# and saves the network data (including node coordinates) to CSV files.

import networkx as nx
import pandas as pd
import os
import numpy as np

# --- 1. Network Configurations ---
NETWORKS_TO_EXPORT = [
    {"name": "Fragmented_1k", "block_sizes": [100] * 10, "p_in": 0.01, "p_out": 0.005},
    {"name": "Core-Periphery_2k", "block_sizes": [1000, 200, 200, 200, 200, 200], "p_in": 0.01, "p_out": 0.005},
    {"name": "Bimodal_3k", "block_sizes": [1500, 1500], "p_in": 0.01, "p_out": 0.005},
    {"name": "Pyramidal_4k", "block_sizes": [1200, 800, 800, 240, 240, 240, 240, 240], "p_in": 0.01, "p_out": 0.005},
    {"name": "SBM_5k_Zero_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.01},
    {"name": "SBM_5k_Low_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.0075},
    {"name": "SBM_5k_Medium_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.005},
    {"name": "SBM_5k_High_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.001},
    {"name": "SBM_5k_Max_Modularity", "block_sizes": [500, 1500, 1000, 800, 1200], "p_in": 0.01, "p_out": 0.0}
]

# --- 2. Custom Layout Calculation Function ---

def get_custom_layout(G, name, block_sizes):
    """
    Calculates x, y coordinates for each node based on the network's conceptual structure.
    """
    block_centers = {}
    
    # --- Define Macro-Layouts for Blocks ---
    if name == "Core-Periphery_2k":
        block_centers[0] = (0, 0) # Core
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        for i in range(5):
            block_centers[i+1] = (15 * np.cos(angles[i]), 15 * np.sin(angles[i]))
            
    elif name == "Pyramidal_4k":
        block_centers[0] = (0, 12)  # Top
        block_centers[1], block_centers[2] = (-6, 4), (6, 4) # Middle
        for i in range(5):
             block_centers[i+3] = (-12 + i * 4.8, -6) # Bottom
             
    elif name == "Bimodal_3k":
        block_centers[0], block_centers[1] = (-8, 0), (8, 0)

    elif name == "Fragmented_1k":
        for i in range(10):
            row = i // 5
            col = i % 5
            block_centers[i] = (col * 10 - 20, -row * 10 + 5)
    else: # Default for all 5k SBMs
        distance = 25 if name == "SBM_5k_Max_Modularity" else 15
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        for i in range(5):
            block_centers[i] = (distance * np.cos(angles[i]), distance * np.sin(angles[i]))

    # --- Calculate Node Positions (Micro-Layouts) ---
    pos = {}
    node_start_index = 0
    for i, size in enumerate(block_sizes):
        nodes_in_block = list(range(node_start_index, node_start_index + size))
        subgraph = G.subgraph(nodes_in_block)
        
        # Use a spring layout for nodes within the block, scaled by block size
        block_pos = nx.spring_layout(subgraph, seed=42, iterations=50)
        
        # Scale and translate node positions to be around the block's center
        scale = np.sqrt(size) / 10
        center_x, center_y = block_centers[i]
        for node, (x, y) in block_pos.items():
            pos[node] = (center_x + x * scale, center_y + y * scale)
        
        node_start_index += size
        
    return pos

# --- 3. Main Execution ---
if __name__ == "__main__":
    output_dir = "network_data_layouts"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data with custom layouts will be saved in '{output_dir}/'.")

    for spec in NETWORKS_TO_EXPORT:
        name = spec["name"]
        print(f"\nProcessing network: {name}...")

        # Generate graph
        G = nx.stochastic_block_model(spec["block_sizes"], 
                                      [[spec["p_out"]] * len(spec["block_sizes"]) for _ in range(len(spec["block_sizes"]))] if spec["p_in"] == spec["p_out"] else \
                                      np.where(np.eye(len(spec["block_sizes"])), spec["p_in"], spec["p_out"]),
                                      seed=42)
        
        # Calculate custom layout
        node_positions = get_custom_layout(G, name, spec["block_sizes"])
        print(f"  > Calculated custom layout coordinates.")

        # Create and save node attributes DataFrame
        node_attributes = []
        node_start_index = 0
        for i, size in enumerate(spec["block_sizes"]):
            for node_idx in range(size):
                node_id = node_start_index + node_idx
                x, y = node_positions[node_id]
                node_attributes.append({"node_id": node_id, "block_id": i + 1, "x": x, "y": y})
            node_start_index += size
        nodes_df = pd.DataFrame(node_attributes)
        
        nodes_filepath = os.path.join(output_dir, f"{name}_nodes.csv")
        nodes_df.to_csv(nodes_filepath, index=False)
        print(f"  > Saved node attributes (with x, y) to {nodes_filepath}")

        # Create and save edge list
        edges_df = nx.to_pandas_edgelist(G, source="source", target="target")
        edges_filepath = os.path.join(output_dir, f"{name}_edges.csv")
        edges_df.to_csv(edges_filepath, index=False)
        print(f"  > Saved edge list to {edges_filepath}")

    print("\n✅ All networks have been successfully exported with custom layouts.")

