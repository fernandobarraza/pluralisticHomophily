"""
Script to process and analyze network data for pluralistic homophily.
Author: Fernando Barraza
Date: Jan 2024

This script loads network data, performs HLC algorithm with various thresholds,
calculates pluralistic homophily, community coverage, overlap coverage, and average scaled link-density,
and outputs the results to CSV files.

Libraries:
- pandas for data manipulation
- argparse for argument parsing
- gc for garbage collection
- hlc for HLC algorithm implementation
- funcs_common for common functions

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Run the script with appropriate arguments for dataset_key, env, factor, and sampling.
"""

import argparse
import gc

import pandas as pd

import hlc
from funcs_common import *


def generate_thresholds(upper_threshold, lower_threshold, total_measurements=10):
    """
    Generates a list of thresholds between upper and lower limits.

    Args:
        upper_threshold (float): The upper limit for thresholds.
        lower_threshold (float): The lower limit for thresholds.
        total_measurements (int): The total number of thresholds to generate.

    Returns:
        list: A list of generated thresholds.
    """
    step = (upper_threshold - lower_threshold) / (total_measurements - 1)
    return [lower_threshold + i * step for i in range(total_measurements)]


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
parser.add_argument('--env', help='Execution environment: local or AWS', default='local')
parser.add_argument('--dataset_key', type=str, help='Key of the dataset to process.', default='TextAb10_wsw')
parser.add_argument('--factor', type=float, help='Percentage factor for sampling.', default=0)
parser.add_argument('--sampling', type=str, help='Sampling technique.', default=None)

# Parse arguments
args = parser.parse_args()
dataset_key = args.dataset_key
factor = args.factor
sampling_method = args.sampling if args.sampling else ''

# Directory configuration based on the environment
if args.env == 'AWS':
    datasets_dir = 'datasets/'
    experiments_dir = 'experiments/' + dataset_key + '/'
    print("Configuration for AWS.")
else:
    datasets_dir = '/Volumes/Doctorado/datasets/ext/'
    experiments_dir = '/Volumes/Doctorado/experiments/ext/results/' + dataset_key + '/'
    print("Configuration for local environment.")

verbose = True
write_flag = True

if factor == 0:
    factor_part = ''
else:
    factor_part = '_' + sampling_method + '_' + str(factor)

network_file_path = f'{datasets_dir}{dataset_key}_network{sampling_method}{factor_part}.txt'

# Graph settings
Is_Directed = False
weight_option = False

print("Running for", dataset_key, "in", args.env)
print(f"Reading ncol file {network_file_path} and building network...")

t0 = time.time()

# Read the graph from the .ncol file
g = Graph.Read_Ncol(network_file_path, weights=weight_option, directed=Is_Directed)

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

# Memory cleanup
gc.collect()

upper_threshold = 0.7
lower_threshold = 0.3

calculated_thresholds = generate_thresholds(upper_threshold, lower_threshold)
print("Thresholds =", calculated_thresholds)

threshold_runs = [None, 0.3, 0.5]

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
    communities_file = f"{experiments_dir}{dataset_key}_{threshold_str}_communities{sampling_method}{factor_part}.txt"

    # Write communities to file
    print("Writing file", communities_file)
    with open(communities_file, "w") as file:
        for community in results:
            file.write("[" + ",".join(g.vs[community]["name"]) + "]\n")
            num_comm_detected += 1

    delay = time.time() - t0
    print("duration: %.2f s." % delay)
    print("Number of raw communities =", num_comm_detected)

    if num_comm_detected > 1:
        print("Calculating pluralistic homophily...")
        h, communities = pluralistic_homophily_alg3(g, communities_file, ',', 'm1')

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
        h_output_file = experiments_dir + 'h/' + dataset_key + '_' + threshold_str + '_homofilies' + sampling_method + factor_part + '.csv'

        if write_flag:
            print("Writing file", h_output_file)
            with open(h_output_file, "w") as f:
                f.write(",".join(atributos) + "\n")
                for vertex in g.vs:
                    valores = [str(vertex[attr]) if attr != 'degree' else str(vertex.degree()) for attr in atributos]
                    f.write(",".join(valores) + "\n")

        delay = time.time() - t0
        print("duration: %.2f s." % delay)

# Save DataFrame to CSV
results_file = f"{experiments_dir}{dataset_key}_networkLevel_results{sampling_method}{factor_part}.csv"
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

delay = time.time() - t0
print("duration: %.2f s." % delay)
