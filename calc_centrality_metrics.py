"""
Script to process and analyze network data for pluralistic homophily.
Author: Fernando Barraza
Date: Jan 2024

This script loads network and homophily data, performs stratified sampling,
calculates centrality measures, and outputs the results to a CSV file.

Libraries:
- pandas for data manipulation
- matplotlib.pyplot for plotting
- numpy for numerical operations
- networkit for graph algorithms

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Run the script with appropriate arguments for dataset_key and env.
"""

import time

import networkit as nk
import pandas as pd

t0 = time.time()
delay = time.time() - t0
print("duration: %.2f s." % delay)


def load_data(graph_path, homophily_path):
    """Loads the graph and pluralistic homophily data."""
    g = read_scc(graph_path)

    try:
        # Attempt to read the file with header
        homophily_data = pd.read_csv(homophily_path, dtype={
            'name': str,
            'h_v': float,
            'degree': int,
            'overlap': int
        })

        # Check if the expected columns are present
        if not all(col in homophily_data.columns for col in ['name', 'h_v', 'degree', 'overlap']):
            raise ValueError("Expected columns are not present. Attempting without header.")

    except (pd.errors.ParserError, ValueError):
        # If an error occurs, attempt to read the file without header
        homophily_data = pd.read_csv(homophily_path, header=None, names=['name', 'h_v', 'degree', 'overlap'], dtype={
            'name': str,
            'h_v': float,
            'degree': int,
            'overlap': int
        })

    return g, homophily_data


def clean_node_ids(ig_graph):
    """Cleans node identifiers in an igraph graph."""
    new_graph = ig_graph.copy()
    for vertex in new_graph.vs:
        cleaned_id = int(vertex['name'].lstrip('DB'))
        vertex['name'] = cleaned_id
    return new_graph


def stratified_sampling(homophily_data, n_samples_per_strata=100, network_size='large'):
    """
    Performs stratified sampling of nodes based on pluralistic homophily.

    Args:
        homophily_data (pd.DataFrame): DataFrame with pluralistic homophily data.
        n_samples_per_strata (int): Number of samples per stratum.
        network_size (str): Size of the network ('small', 'medium', 'large').

    Returns:
        tuple: Lists of sampled nodes, homophily values, and labels.
    """
    mean_h_v = homophily_data['h_v'].mean()
    std_h_v = homophily_data['h_v'].std()

    if network_size == 'small':  # Biological networks
        threshold = std_h_v / 4
    elif network_size == 'medium':  # Informational networks
        threshold = std_h_v / 2
    else:  # Large networks (social)
        threshold = std_h_v

    non_assortative_low = mean_h_v - threshold
    non_assortative_high = mean_h_v + threshold

    assortative = homophily_data[homophily_data['h_v'] > non_assortative_high]
    non_assortative = homophily_data[
        (homophily_data['h_v'] >= non_assortative_low) & (homophily_data['h_v'] <= non_assortative_high)]
    disassortative = homophily_data[homophily_data['h_v'] < non_assortative_low]

    print("Len of stratified samples:")
    print('A=', len(assortative))
    print('Non-A=', len(non_assortative))
    print('D=', len(disassortative))

    sampled_nodes = []
    homophily_values = []
    labels = []

    for df, label in [(assortative, 'A'), (non_assortative, 'Non-A'), (disassortative, 'D')]:
        sampled = df.sample(n=min(n_samples_per_strata, len(df)), replace=len(df) < n_samples_per_strata)
        sampled_nodes.extend(sampled['name'].tolist())
        homophily_values.extend(sampled['h_v'].tolist())
        labels.extend([label] * len(sampled))

    return sampled_nodes, homophily_values, labels


def ig_to_nk(ig_graph):
    """Converts an igraph graph to a NetworKit graph and creates an index-to-name mapping."""
    nk_graph = nk.Graph(n=ig_graph.vcount(), directed=ig_graph.is_directed())
    index_to_name = {}
    for vertex in ig_graph.vs:
        nk_index = nk_graph.addNode()
        index_to_name[vertex.index] = nk_index

    for vertex in ig_graph.vs:
        nk_index = index_to_name[vertex.index]
        for neighbor in vertex.neighbors():
            neighbor_nk_index = index_to_name[neighbor.index]
            if not nk_graph.hasEdge(nk_index, neighbor_nk_index):
                nk_graph.addEdge(nk_index, neighbor_nk_index)

    return nk_graph, index_to_name


def calculate_centrality(graph, nodes):
    """Calculates centrality metrics for the given nodes in the graph."""
    nk_graph, index_to_name = ig_to_nk(graph)
    node_indices = [graph.vs.find(name=str(node)).index for node in nodes if graph.vs.find(name=str(node)) is not None]

    print("Indexes nodes (count):", len(node_indices))
    tracker = TimeTracker()

    print("Calculating degree centrality...")
    degree = [graph.degree(v) / (graph.vcount() - 1) for v in node_indices]
    tracker.print_time()

    print("Calculating closeness centrality...")
    closeness = graph.closeness(vertices=node_indices, normalized=True)
    tracker.print_time()

    print("Calculating eigenvector centrality...")
    eigenvector = graph.eigenvector_centrality(scale=True)
    eigenvector = [eigenvector[i] for i in node_indices]
    tracker.print_time()

    print("Calculating betweenness centrality...")
    betweenness_calculator = nk.centrality.ApproxBetweenness(nk_graph, epsilon=0.1)
    betweenness_calculator.run()
    betweenness_scores = betweenness_calculator.scores()
    tracker.print_time()

    betweenness = [betweenness_scores[index_to_name[i]] for i in node_indices]
    final_node_names = [graph.vs[i]['name'] for i in node_indices]
    print('Total nodes processed = ', len(final_node_names))

    result = {
        'node': final_node_names,
        'degree': degree,
        'closeness': closeness,
        'eigenvector': eigenvector,
        'betweenness': betweenness
    }
    return pd.DataFrame(result)


def calculate_network_metric(graph_path, homophily_path, output_path, node_sample_size, network_key):
    """Calculates and saves network metrics for the given graph and homophily data."""
    g, homophily_data = load_data(graph_path, homophily_path)

    if network_key in ('so', 'dblp', 'amazon'):
        network_size = 'medium'
    elif network_key in ('ppi', 'ddi', 'celegans'):
        network_size = 'small'
    else:
        network_size = 'large'

    sampled_nodes, homophily_values, labels = stratified_sampling(homophily_data, node_sample_size, network_size)
    print("graph node count =", g.vcount())
    print("graph edge count =", g.ecount())
    print("Node sample size =", node_sample_size)

    centrality_df = calculate_centrality(g, sampled_nodes)
    homophily_label_df = pd.DataFrame({
        'node': sampled_nodes,
        'homophily_value': homophily_values,
        'label': labels
    })

    centrality_df['node'] = centrality_df['node'].astype(str)
    homophily_label_df['node'] = homophily_label_df['node'].astype(str)
    final_df = centrality_df.merge(homophily_label_df, on='node', how='left')
    final_df = final_df.drop_duplicates(subset=['node'], keep='first')
    final_df.to_csv(output_path, index=False)
    print(f"Centralities written to {output_path}")


if __name__ == "__main__":
    from funcs_common import *
    import argparse

    parser = argparse.ArgumentParser(description='Script for data processing.')
    parser.add_argument('--env', help='Execution environment: local or AWS', default='local')
    parser.add_argument('--dataset_key', type=str, help='Key of the dataset to process.', default='TextAb10_wsw')
    args = parser.parse_args()

    base_dir = '/Volumes/Doctorado/'
    networks = {
        'so': (None, 'StackOverflow'),
        'dblp': (None, 'DBLP'),
        'amazon': (None, 'Amazon'),
        'livejournal': (0.05, 'LiveJournal'),
        'youtube': (0.1, 'YouTube'),
        'orkut': (0.01, 'Orkut'),
        'ppi': (None, 'Protein-Protein Interaction'),
        'ddi': (None, 'Drug-Drug Interaction'),
        'celegans': (None, 'C. Elegans')
    }

    test_network = {
        'celegans': (None, 'C. Elegans')
    }

    networks = test_network
    node_sample_size = 3000
    datasets_dir = base_dir + 'datasets/ext/'

    for i, (network_key, value) in enumerate(networks.items()):
        print('Running for ', i, network_key, value)
        factor = value[0]
        factor_part = f'_RWR_{factor}' if factor else ''
        experiments_dir = base_dir + 'experiments/ext/'
        network_file_path = f'{datasets_dir}{network_key}_network{factor_part}.txt'
        print("Network file at", network_file_path)

        if factor:
            communities_file = f'{experiments_dir}results/{network_key}/{network_key}_def_communities{factor_part}.txt'
            homophily_path = f'{experiments_dir}results/{network_key}/h/{network_key}_def_homofilies{factor_part}.csv'
        else:
            communities_file = f'{datasets_dir}{network_key}_communities{factor_part}.txt'
            homophily_path = experiments_dir + network_key + '_homophilies_alg3_m1.csv'

        print("Community file at", communities_file)
        print("Homophily file at", homophily_path)
        output_path = f'{experiments_dir}node_level_analysis/{network_key}_node-level_metrics_{str(node_sample_size)}.csv'
        print("Output file at ", output_path)

        calculate_network_metric(network_file_path, homophily_path, output_path, node_sample_size, network_key)
