"""
Script to calculate pluralistic homophily and extended metrics.
Author: Fernando Barraza
Date: Jan 2024

This script loads network data, performs HLC algorithm with various thresholds,
calculates pluralistic homophily, community coverage, overlap coverage, and average scaled link-density,
and outputs the results to CSV files.

Libraries:
- pandas for data manipulation
- argparse for argument parsing
- hlc for HLC algorithm implementation
- funcs_common for common functions

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
"""

import argparse

import pandas as pd

import hlc
from funcs_common import *

def calcular_densidad_escalada_promedio(grafo, comunidades):
    """
    Calculates the average scaled density of a graph based on its communities.

    Args:
        grafo (Graph): The graph.
        comunidades (list): List of communities.

    Returns:
        float: The average scaled density of the graph.
    """
    densidades_escaladas = []
    for comunidad in comunidades:
        subgrafo = grafo.subgraph(comunidad)
        t = subgrafo.ecount()
        s = len(comunidad)

        if s > 1:
            rho = 2 * t / (s * (s - 1))
            rho_tilde = rho * s
            densidades_escaladas.append(rho_tilde)
        else:
            densidades_escaladas.append(0)

    if densidades_escaladas:
        densidad_escalada_promedio = sum(densidades_escaladas) / len(densidades_escaladas)
    else:
        densidad_escalada_promedio = 0

    return densidad_escalada_promedio


# Argument parser configuration
parser = argparse.ArgumentParser(description='Script for data processing.')
parser.add_argument('--dataset_file', type=str, help='Name of dataset file to process.', default='TextAb10_wsw')

# Parse arguments
args = parser.parse_args()
dataset_file = args.dataset_name

# Graph settings
Is_Directed = False
weight_option = False

print(f"Reading ncol file {dataset_file} and building network...")

t0 = time.time()

# Read the graph from the .ncol file
g = Graph.Read_Ncol(dataset_file, weights=weight_option, directed=Is_Directed)

delay = time.time() - t0
print("duration: %.2f s." % delay)

print("Num of vertices =", g.vcount())
print("Number of edges =", g.ecount())

# Get the largest connected component
c = g.connected_components("strong")
sccIdx = c.sizes().index(max(c.sizes()))
scc = c.subgraph(sccIdx)

# Memory cleanup
del c, sccIdx

# Simplify the graph by combining parallel edges
scc.simplify(combine_edges={"weight": sum})

print("Num of vertices (scc) =", scc.vcount())
print("Number of edges (scc) =", scc.ecount())

g = scc

# Initialize the HLC algorithm with the graph
alg = hlc.HLC(g)

# Define thresholds for the HLC algorithm
threshold_runs = [None]
#threshold_runs = [None, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Create DataFrame to store results
results_df = pd.DataFrame(
    columns=['Threshold', 'Pluralistic Homophily', 'Community Coverage', 'Overlap Coverage', 'Avg Scaled Link-Density'])

# Execute the HLC algorithm and store results
for threshold in threshold_runs:
    print("\nResults for threshold =", threshold)
    print("Executing HLC ...")

    if threshold:
        results = alg.run(threshold)
    else:
        results = alg.run()

    delay = time.time() - t0
    print("duration: %.2f s." % delay)
    print("last partition density =", alg.last_partition_density)

    num_comm_detected = 0
    threshold_str = f"{threshold:.2f}" if threshold is not None else 'def'

    # Write communities to file
    community_file = 'community_'+threshold_str+'.txt'
    print("Writing file", community_file)
    with open(community_file, "w") as file:
        for community in results:
            file.write("[" + ",".join(g.vs[community]["name"]) + "]\n")
            num_comm_detected += 1

    delay = time.time() - t0
    print("duration: %.2f s." % delay)
    print("Number of raw communities =", num_comm_detected)

    if num_comm_detected > 1:
        print("Calculating pluralistic homophily...")
        h, communities = pluralistic_homophily(g, community_file, ',', 'm1')

        delay = time.time() - t0
        print("duration: %.2f s." % delay)
        print("Number of non-trivial communities =", len(communities))
        print("Network pluralistic homophily:", h)

        CC = community_coverage(g, communities)
        OC = overlap_coverage(g, communities)
        densidad_promedio = calcular_densidad_escalada_promedio(g, communities)
        print("Average scaled link-density of the network:", densidad_promedio)
        print("Community coverage =", CC)
        print("Overlap coverage =", OC)

        delay = time.time() - t0
        print("duration: %.2f s." % delay)

        # Add row to DataFrame
        new_row = pd.DataFrame({'Threshold': [threshold_str], 'Pluralistic Homophily': [h], 'Community Coverage': [CC],
                                'Overlap Coverage': [OC], 'Avg Scaled Link-Density': [densidad_promedio]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        atributos = ['name', 'h_v', 'degree', 'overlap']
        h_output_file = 'homofilies_'+threshold_str + '.csv'

        # Writting local pluralistic homophilies
        print("Writing file", h_output_file)
        with open(h_output_file, "w") as f:
            f.write(",".join(atributos) + "\n")
            for vertex in g.vs:
                valores = [str(vertex[attr]) if attr != 'degree' else str(vertex.degree()) for attr in atributos]
                f.write(",".join(valores) + "\n")

        delay = time.time() - t0
        print("duration: %.2f s." % delay)

# Save DataFrame to CSV
results_file = f"networkLevel_results.csv"
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

delay = time.time() - t0
print("duration: %.2f s." % delay)
