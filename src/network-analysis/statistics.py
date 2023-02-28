"""This module contains classes and functions to compute statistics on
networks.
"""
import networkx as nx
import numpy as np
from scipy.special import comb


def gwd(graph, decay):
    """Compute the geometrically weighted degree of the simple graph.

    :param graph: The graph.
    :type graph: NetworkX simple graph.
    :param decay: The decay parameter.
    :type decay: Positive float.
    :return: The geometrically weighted degree.
    :rtype: Float.
    """
    degrees = np.array([d for _, d in graph.degree()])
    uniques, counts = np.unique(degrees, return_counts=True)
    weighted_degrees = (1 - (1 - np.exp(-decay)) ** uniques) * counts
    return np.exp(decay) * weighted_degrees.sum()


def gwesp(graph, decay):
    """Compute the geometrically weighted edgewise shared partners of the
    graph.

    :param graph: The graph.
    :type graph: NetworkX simple graph.
    :param decay: The decay parameter.
    :type decay: Positive float.
    :return: The geometrically weighted edgewise shared partners.
    :rtype: Float.
    """
    adj_matrix = nx.to_numpy_array(graph)
    n_common_neighbours = (adj_matrix @ adj_matrix) * adj_matrix
    upper_diag_idx = np.triu_indices(adj_matrix.shape[0], 1)
    uniques, counts = np.unique(
        n_common_neighbours[upper_diag_idx], return_counts=True
    )
    weighted_ew_shared_partners = (
        1 - (1 - np.exp(-decay)) ** uniques
    ) * counts
    return np.exp(decay) * weighted_ew_shared_partners.sum()


def kstars(graph, k):
    """Count the number of k-stars of the undirected graph.

    :param graph: The graph.
    :type graph: NetworkX undirected graph.
    :param k: The number of branch of a star.
    :type k: Integer strictly greater than 0.
    :return: The number of k-stars the graph contains.
    :rtype: Integer.
    """
    degrees = np.array([d for _, d in graph.degree()])
    return comb(degrees, k).sum()


def in_kstars(graph, k):
    """Count the number of in k-stars of the directed graph. An in k-star in
    composed of arcs pointing towards its center.

    :param graph: The graph.
    :type graph: NetworkX directed graph.
    :param k: The number of branch of a star.
    :type k: Integer strictly greater than 0.
    :return: The number of in k-stars the graph contains.
    :rtype: Integer.
    """
    degrees = np.array([d for _, d in graph.in_degree()])
    return comb(degrees, k).sum()


def out_kstars(graph, k):
    """Count the number of out k-stars of the directed graph. An out k-star in
    composed of arcs pointing towards its border.

    :param graph: The graph.
    :type graph: NetworkX directed graph.
    :param k: The number of branch of a star.
    :type k: Integer strictly greater than 0.
    :return: The number of out k-stars the graph contains.
    :rtype: Integer.
    """
    degrees = np.array([d for _, d in graph.out_degree()])
    return comb(degrees, k).sum()


def mutuals(graph):
    """Count the number of pairs of nodes in the graph that has a mutual
    connection. In a undirected multigraph, two nodes have a mutual connection
    if there are at least two edges between them. In a directed graph, two arcs
    between two nodes must be of opposite direction for having a mutual
    connection.

    :param graph: The graph.
    :type graph: NetworkX graph.
    :return: The number of mutual connections.
    :rtype: Integer.
    """
    count = 0
    for u, v in graph.edges():
        if graph.has_edge(v, u):
            count += 1
    return count // 2


class Statistics:
    """Base class for statistics about graphs."""

    def __call__(self, graph):
        raise NotImplementedError(
            "The __call__ special method of a statistics must be implemented."
        )
