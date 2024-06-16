import collections
import math
import time

import numpy as np
from igraph import Graph


class TimeTracker:
    """Tracks and prints the elapsed time for processes."""

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def check_time(self):
        """Checks the time elapsed since the last check and since the start."""
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        since_last_check = current_time - self.last_time
        self.last_time = current_time
        return total_elapsed / 60, since_last_check / 60  # Return times in minutes

    def print_time(self):
        """Prints the total elapsed time and time since the last check."""
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        since_last_check = current_time - self.last_time
        self.last_time = current_time
        total = total_elapsed / 60
        since_last = since_last_check / 60
        print(f"Total time elapsed: {total:.2f} minutes, Time since last check: {since_last:.2f} minutes")


def read_scc(network_file_path, weights_option=False, direction=False):
    """Reads the network file and extracts the largest strongly connected component."""
    g = Graph.Read_Ncol(network_file_path, weights=weights_option, directed=direction)
    c = g.connected_components("strong")
    sccIdx = c.sizes().index(max(c.sizes()))
    scc = c.subgraph(sccIdx)
    del c, sccIdx
    scc.simplify(combine_edges={"weight": sum})
    return scc


def calculate_sample_size_to_plot(sample_size):
    """Calculates the sample size to plot based on the total sample size."""
    if sample_size <= 1000:
        return sample_size
    else:
        exponent = math.floor(math.log10(sample_size)) - 1
        return max(1000, 10 ** exponent)


def assign_scalar_values(scc, communities, method, attribute='scalar_value'):
    """Assigns scalar values to nodes based on the specified method."""
    node_communities = {str(vertex['name']): set() for vertex in scc.vs}

    for i, community in enumerate(communities):
        for node_name in community:
            node_communities[str(node_name)].add(i)

    if method == 'm1':
        for vertex in scc.vs:
            vertex_name = vertex['name']
            vertex[attribute] = len(node_communities[vertex_name])
    elif method == 'm2':
        for vertex in scc.vs:
            vertex_name = vertex['name']
            shared_neighbors_count = sum(
                len(node_communities[vertex_name].intersection(node_communities[scc.vs[nei]['name']])) > 0
                for nei in scc.neighbors(vertex.index)
            )
            vertex[attribute] = shared_neighbors_count
    elif method == 'm3':
        for vertex in scc.vs:
            vertex_name = vertex['name']
            total_shared_communities = sum(
                len(node_communities[vertex_name].intersection(node_communities[scc.vs[nei]['name']]))
                for nei in scc.neighbors(vertex.index)
            )
            vertex[attribute] = total_shared_communities
    elif method == 'm4':
        for vertex in scc.vs:
            vertex_name = vertex['name']
            if vertex.degree() > 0:
                avg_shared_communities = sum(
                    len(node_communities[vertex_name].intersection(node_communities[scc.vs[nei]['name']]))
                    for nei in scc.neighbors(vertex.index)) / vertex.degree()
                vertex[attribute] = round(avg_shared_communities)
            else:
                vertex[attribute] = 0
    elif method == 'm5':
        for vertex in scc.vs:
            vertex_name = vertex['name']
            if vertex.degree() > 0:
                avg_shared_communities = sum(
                    len(node_communities[vertex_name].intersection(node_communities[scc.vs[nei]['name']]))
                    for nei in scc.neighbors(vertex.index)) / vertex.degree()
                vertex[attribute] = avg_shared_communities
            else:
                vertex[attribute] = 0


def read_file(file, separator):
    """Reads a file and processes its lines."""
    result_list = []
    with open(file, "r") as f:
        for line in f:
            line2 = ''.join(c for c in line if c not in '[]').strip()
            list_line = [element.strip() for element in line2.split(separator)]
            result_list.append(list_line)
    return result_list


def read_overlaps(communities_file, scc, file_sep=',', non_trivial=3, scalar_method='m1'):
    """Reads community overlaps from a file and assigns scalar values."""
    communities = read_file(communities_file, file_sep)
    non_trivial_communities = [i for i in communities if len(i) > non_trivial]
    assign_scalar_values(scc, communities, scalar_method, 'overlap')

    for v in scc.vs:
        if v["overlap"] is None:
            scc.vs[v.index]["overlap"] = 0

    return non_trivial_communities


def pluralistic_homophily_alg3(g, comm_file, comm_file_sep, scalar_method):
    """Calculates pluralistic homophily using Algorithm 3."""

    def p_k(k, degrees):
        return degrees.count(k) / len(degrees)

    def q_k(k, degrees):
        return (k + 1) * p_k(k + 1, degrees) / np.average(degrees)

    communities = read_overlaps(comm_file, g, file_sep=comm_file_sep, non_trivial=0, scalar_method=scalar_method)

    nodes_overlap = [g.vs[vertex.index]["overlap"] for vertex in g.vs]
    graph_overlaps = nodes_overlap
    maximo = math.ceil(max(graph_overlaps))

    excess_overlap_dist = [0] * int(maximo)
    for j in range(maximo):
        excess_overlap_dist[j] = q_k(j, graph_overlaps)

    mu_q = sum(i * excess_overlap_dist[i] for i in range(maximo))
    V_q = sum(i * i * excess_overlap_dist[i] for i in range(maximo))
    V_q -= mu_q ** 2

    M = g.ecount()
    denominator = (2 * M * V_q)
    total_network_assortativity = 0

    for vertex in g.vs:
        vertex_overlap = g.vs[vertex.index]["overlap"] - 1
        neis_overlap = [g.vs[neigh]["overlap"] for neigh in g.neighbors(vertex.index)]

        numerator = sum((vertex_overlap - mu_q) * ((neigh_overlap - 1) - mu_q) for neigh_overlap in neis_overlap)
        h_v = numerator / denominator

        g.vs[vertex.index]["h_v"] = h_v
        total_network_assortativity += h_v

    igraph_assortativity = g.assortativity(types1="overlap", types2=None, directed=False)
    alpha = igraph_assortativity / total_network_assortativity
    new_sum_h = 0

    for vertex in g.vs:
        new_h_v = alpha * g.vs[vertex.index]["h_v"]
        new_sum_h += new_h_v

    return new_sum_h, communities


def overlap_coverage(g, communities):
    """Calculates the overlap coverage of communities."""
    non_trivial_communities = [i for i in communities if len(i) > 2]
    vertexes_flat_list = [item for sublist in non_trivial_communities for item in sublist]
    counter_vertexes = collections.Counter(vertexes_flat_list)

    overlap_sum = sum(int(vertex[1]) for vertex in counter_vertexes.most_common())
    all_users_graph_count = g.vcount()

    return overlap_sum / all_users_graph_count


def community_coverage(g, communities):
    """Calculates the community coverage of the graph."""
    all_users_graph_count = g.vcount()
    all_users_comm_count = len(set().union(*communities))
    return all_users_comm_count / all_users_graph_count
