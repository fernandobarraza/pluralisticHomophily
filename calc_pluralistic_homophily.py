"""
Script to process and analyze network data for pluralistic homophily.
Author: Fernando Barraza
Date: Jan 2024

This script processes network data to calculate pluralistic homophily, community coverage, overlap coverage,
and other network metrics. The results are output to the console and optionally to CSV files.

Libraries:
- pandas for data manipulation
- numpy for numerical operations
- funcs_common for common functions

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Run the script with appropriate arguments for dataset, environment, and other parameters.
"""

import math
import time

import numpy as np
from igraph import Graph

from funcs_common import read_overlaps

# Set verbose and write flag options
verbose = True
write_flag = False

# Define dataset
dataset = 'celegans'

if verbose:
    print("Dataset: ", dataset)

# Define directories and file extensions
input_directory = '/Volumes/Doctorado/'
output_directory = '/Volumes/Doctorado/experiments/'
proc_dir = 'ext/'
data_dir = 'datasets/ext/'
comm_dir = data_dir
file_ext = ".txt"
comm_file_sep = '\t'
weights_option = False

scalar_method = 'm1'

# Define dataset-specific parameters
if dataset == 'SO':
    target_graph = 'SO'
    weights_option = True
    annotation_file = "UniqueUserTags"
    input_file = "so_network"
    comm_input_file = "so_communities"
    comm_file_sep = ' '

elif dataset == 'DBLP':
    target_graph = 'DBLP'
    annotation_file = "AuthorJournals"
    input_file = 'dblp_network'
    comm_input_file = "dblp_communities"

elif dataset == 'Amazon':
    target_graph = 'Amazon'
    annotation_file = "ProductCategories"
    input_file = 'amazon_network'
    comm_input_file = "amazon_communities"

elif dataset == 'Youtube':
    target_graph = 'Youtube'
    annotation_file = "YouUsrGroups"
    input_file = 'youtube_network'
    comm_input_file = "youtube_communities"

elif dataset == 'Orkut':
    target_graph = 'Orkut'
    annotation_file = "OrkutGroups"
    input_file = 'orkut_network'
    comm_input_file = "orkut_communities"

elif dataset == 'Livejournal':
    target_graph = 'Livejournal'
    annotation_file = "UserGroups"
    input_file = 'livejournal_network'
    comm_input_file = "livejournal_communities"

elif dataset == 'ppi':
    target_graph = 'ppi'
    annotation_file = ""
    input_file = 'ppi_network'
    comm_input_file = "ppi_communities"
    comm_file_sep = ','

elif dataset == 'ddi':
    target_graph = 'ddi'
    annotation_file = ""
    input_file = 'ddi_network'
    comm_input_file = "ddi_communities"
    comm_file_sep = ','
    file_ext = '.tsv'

elif dataset == 'celegans':
    target_graph = 'celegans'
    annotation_file = ""
    input_file = 'celegans_network'
    comm_input_file = "celegans_communities"
    comm_file_sep = ','
    file_ext = '.txt'

elif dataset == 'TextAb':
    comm_file_sep = ','
    weights_option = True
    input_file = 'TextAb10_wsw_network'
    target_graph = 'TextAb10_wsw'
    file_ext = ".txt"
    annotation_file = ""
    comm_dir = 'experiments/ext/'
    comm_input_file = target_graph + "_communities"

else:
    input_file = 'toy_example_network'
    input_directory = './'
    output_directory = './'
    data_dir = ''
    comm_dir = ''
    proc_dir = ''
    target_graph = 'toy'
    comm_input_file = "toy_example_communities"
    comm_file_sep = ','
    file_ext = '.txt'

output_file = target_graph + "_homophilies_alg3_" + scalar_method

print("Reading ncol file and building network...")
t0 = time.time()

# Read the graph from the .ncol file
g = Graph.Read_Ncol(input_directory + data_dir + input_file + file_ext, weights=weights_option, directed=False)

print(f"Duration: {time.time() - t0:.2f} s")
print(f"Num of vertices of initial graph = {g.vcount()}")
print(f"Number of edges of initial graph = {g.ecount()}")

print("Calculating LCC...")
c = g.clusters("strong")

sccIdx = c.sizes().index(max(c.sizes()))
scc = c.subgraph(sccIdx)
scc.simplify(combine_edges={"weight": sum})

print(f"Num of vertices of simplified graph = {scc.vcount()}")
print(f"Number of edges of simplified graph = {scc.ecount()}")

average_degree = np.mean(scc.degree())

comm_file = input_directory + comm_dir + comm_input_file + ".txt"
print("Reading community file...")
communities = read_overlaps(comm_file, scc, file_sep=comm_file_sep, non_trivial=0, scalar_method=scalar_method)

num_comms = len(communities)
community_memberships = [len(comm) for comm in communities]
average_memberships = np.mean(community_memberships)


def p_k(k, degrees):
    return degrees.count(k) / len(degrees)


def q_k(k, degrees):
    return (k + 1) * p_k(k + 1, degrees) / np.average(degrees)


nodes_overlap = [v["overlap"] for v in scc.vs]
graph_overlaps = nodes_overlap
maximo = math.ceil(max(graph_overlaps))

excess_overlap_dist = [q_k(j, graph_overlaps) for j in range(maximo)]

mu_q = sum(i * excess_overlap_dist[i] for i in range(maximo))
V_q = sum(i * i * excess_overlap_dist[i] for i in range(maximo)) - mu_q ** 2

N = scc.ecount()
denominator = 2 * N * V_q

print(f"N = {N}")
print(f"mu = {mu_q}")
print(f"Variance = {V_q}")
print(f"Denominator = {denominator}")

total_network_assortativity = 0

print("Calculating node assortativity...")

for vertex in scc.vs:
    vertex_overlap = vertex["overlap"] - 1
    neis_overlap = [scc.vs[neigh]["overlap"] for neigh in scc.neighbors(vertex.index)]
    numerator = sum((vertex_overlap - mu_q) * (neigh_overlap - 1 - mu_q) for neigh_overlap in neis_overlap)
    h_v = numerator / denominator
    vertex["h_v"] = h_v
    total_network_assortativity += h_v

print(f"Sum of node assortativities = {total_network_assortativity}")

igraph_overlap_assortativity = scc.assortativity(types1="overlap", types2=None, directed=False)
print(f"igraph overlap assortativity = {igraph_overlap_assortativity}")

alpha = igraph_overlap_assortativity / total_network_assortativity

new_sum_h = 0
if write_flag:
    with open(output_directory + proc_dir + output_file + ".csv", "w") as f:
        for vertex in scc.vs:
            new_h_v = alpha * vertex["h_v"]
            new_sum_h += new_h_v
            f.write(f"{vertex['name']},{new_h_v},{vertex.degree()},{vertex['overlap']}\n")
else:
    for vertex in scc.vs:
        new_h_v = alpha * vertex["h_v"]
        new_sum_h += new_h_v

print(f"Values scaled with alpha = {alpha}:")
print(f"Sum of node assortativities = {new_sum_h}")
print(f"igraph assortativity = {igraph_overlap_assortativity}")

igraph_assortativity_degree = scc.assortativity_degree()

# Print basic network properties for the table
print(f"Dataset: {dataset}")
print(f"Number of nodes (N): {scc.vcount()}")
print(f"Number of edges (E): {scc.ecount()}")
print(f"Average node degree (<d>): {average_degree}")
print(f"Number of communities (M): {num_comms}")
print(f"Average number of community memberships per node (<m>): {average_memberships}")
print(f"Pluralistic homophily coefficient (h): {new_sum_h}")
print(f"Degree assortativity coefficient (r): {igraph_assortativity_degree}")

print(f"Total duration: {time.time() - t0:.2f} s")
