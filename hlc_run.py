"""
Script for Network Analysis and Community Detection
Author: Fernando Barraza
Date: Aug 2023

This script reads a network file, extracts the largest strongly connected component (SCC),
simplifies the graph, and performs community detection using the HLC algorithm.
The results are saved to an output file.

Libraries:
- igraph for graph manipulation
- hlc for hierarchical link clustering (assuming you have this library)
- gc for garbage collection

Usage:
Modify the `network_file_path` to point to the appropriate network file.
Run the script to perform community detection at various thresholds and save the results.
"""

import sys

from igraph import *

# General settings
verbose = True
experiments_dir = '/Volumes/Doctorado/experiments/ext/'
datasets_dir = '/Volumes/Doctorado/datasets/ext/'
dataset_key = 'celegans'

# Path to the network file
network_file_path = f'{datasets_dir}{dataset_key}_network.txt'

# Graph properties
Is_Directed = False
weight_option = True

print(f"Reading ncol file {network_file_path} and building network...")

# Read the graph from the .ncol file
g = Graph.Read_Ncol(network_file_path, weights=weight_option, directed=Is_Directed)

# Obtain the largest strongly connected component
c = g.connected_components("strong")
sccIdx = c.sizes().index(max(c.sizes()))
scc = c.subgraph(sccIdx)

# Clean up memory
del c, sccIdx

# Simplify the graph by combining parallel edges
scc.simplify(combine_edges={"weight": sum})

# Print the number of vertices and edges
print("Number of vertices =", scc.vcount())
print("Number of edges =", scc.ecount())

# Exit the script early for testing purposes (remove this line in the final script)
sys.exit()

# Assign the simplified graph to the original variable
g = scc

# Initialize the HLC algorithm with the graph
alg = hlc.HLC(g)

# Clean up memory
gc.collect()

# Define thresholds for the HLC algorithm
threshold_runs = [None, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Execute the HLC algorithm and store the results
for threshold in threshold_runs:
    print("Threshold =", threshold)

    if threshold is not None:
        results = alg.run(threshold)
    else:
        results = alg.run()

    # Counter for the number of communities
    j = 0

    # Output file for communities
    communities_file = f'{experiments_dir}{dataset_key}_{str(threshold)}_communities.txt'

    # Write communities to the output file
    print("Writing file", communities_file)
    with open(communities_file, "w") as file:
        for community in results:
            file.write("[" + ",".join(g.vs[community]["name"]) + "]\n")
            j += 1

    # Print the number of communities and algorithm statistics
    print("Number of communities =", j)
    print(f"Threshold = {alg.last_threshold:.6f}")
    print(f"Partition Density = {alg.last_partition_density}")
