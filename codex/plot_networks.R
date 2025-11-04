# plot_custom_layouts.R
# Description: Reads network data with pre-calculated coordinates and generates
# high-quality, custom layout plots using ggraph's manual layout feature.

# --- 1. Setup ---
# install.packages(c("tidyverse", "igraph", "ggraph"))
library(tidyverse)
library(igraph)
library(ggraph)

# --- 2. Configuration ---
data_dir <- "network_data_layouts" # Directory with new layout files
plots_dir <- "network_plots_custom"   # Save to a new directory

if (!dir.exists(plots_dir)) {
  dir.create(plots_dir)
}

# --- 3. Plotting Loop ---
node_files <- list.files(data_dir, pattern = "_nodes\\.csv$", full.names = TRUE)

for (node_file_path in node_files) {
  
  network_name <- str_replace(basename(node_file_path), "_nodes\\.csv$", "")
  edge_file_path <- file.path(data_dir, paste0(network_name, "_edges.csv"))
  
  cat("\nProcessing:", network_name, "...\n")
  
  # --- 4. Data Loading and Graph Creation ---
  nodes_df <- read_csv(node_file_path, show_col_types = FALSE)
  edges_df <- read_csv(edge_file_path, show_col_types = FALSE)
  
  # Create graph object, igraph will automatically pick up x and y as vertex attributes
  graph_obj <- graph_from_data_frame(d = edges_df, vertices = nodes_df, directed = FALSE)
  
  # Differentiate edge types for styling
  E(graph_obj)$edge_type <- E(graph_obj) %>%
    as_ids() %>%
    str_split("\\|") %>%
    map_chr( ~ {
      node1_id <- .x[1]
      node2_id <- .x[2]
      block1 <- V(graph_obj)$block_id[V(graph_obj)$name == node1_id]
      block2 <- V(graph_obj)$block_id[V(graph_obj)$name == node2_id]
      ifelse(block1 == block2, "Intra-Block", "Inter-Block")
    })
  
  cat(" > Graph created with pre-calculated x, y coordinates.\n")
  
  # --- 5. Visualization with a MANUAL ggraph Layout ---
  
  # THE KEY CHANGE: Create a layout object directly from the x and y columns.
  manual_layout <- create_layout(graph_obj, layout = 'manual', 
                                 x = V(graph_obj)$x, y = V(graph_obj)$y)
  
  # Now, create the plot using this pre-defined layout
  network_plot <- ggraph(manual_layout) + 
    geom_edge_link(aes(edge_linetype = edge_type, color = edge_type), 
                   edge_width = 0.3, alpha = 0.5) +
    scale_edge_linetype_manual(values = c("Intra-Block" = "solid", "Inter-Block" = "dashed")) +
    scale_edge_color_manual(values = c("Intra-Block" = "grey40", "Inter-Block" = "grey70")) +
    
    geom_node_point(aes(color = as.factor(block_id)), size = 1.5, alpha = 0.7) +
    
    # Use a colorblind-friendly palette for the nodes
    scale_color_brewer(palette = "Paired") +
    
    theme_graph(base_family = 'sans') +
    labs(
      title = paste("Custom Layout:", network_name),
      subtitle = "Node positions are pre-calculated to show conceptual structure. Dashed lines are inter-block links.",
      caption = "Visualization: ggraph",
      color = "Block ID"
    ) +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 10)
    ) +
    guides(
      edge_linetype = "none",
      edge_color = "none",
      color = guide_legend(override.aes = list(size = 5))
    )
  
  # --- 6. Save the Plot ---
  output_filename <- file.path(plots_dir, paste0(network_name, "_custom_layout.pdf"))
  
  ggsave(
    output_filename, 
    plot = network_plot, 
    width = 11, 
    height = 8.5, 
    units = "in",
    device = 'pdf'
  )
  
  cat(" > Custom layout plot saved to:", output_filename, "\n")
}

cat("\n✅ All custom network plots have been successfully generated.\n")

