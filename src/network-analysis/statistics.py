"""This module contains classes and functions to compute statistics on
networks.
"""
import networkx as nx
import numpy as np


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
    """Count the number of k-stars of the undirected graph. If a multigraph is
    passed as arguement, then multiple edges between two nodes will be
    considered as one.

    :param graph: The graph.
    :type graph: NetworkX undirected graph.
    :param k: The number of branch of a star.
    :type k: Integer strictly greater than 0.
    :return: The number of k-stars the graph contains.
    :rtype: Integer.
    """
    pass


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
    pass


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
    pass


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
    pass


def kcycles(graph, k):
    """Count the number of cycles of length k in the graph.

    :param graph: The graph.
    :type graph: NetworkX graph.
    :param k: The length of a cycle.
    :type k: Integer strictly greater than 1.
    :return: The number of k-cycles.
    :rtype: Integer.
    """
    pass


def transitiveties(graph, k=2):
    """The number of transitivety connection in the directed graph.

    :param graph: The graph.
    :type graph: NetworkX directed graph.
    :param k: The maximal length of the transitity connection.
    :type k: Integer strictly greater than 1.
    :return: The number of transitivety connections.
    :rtype: Integer.
    """
    pass


def diam(graph):
    """The diameter of the graph.

    :param graph: The graph.
    :type graph: NetworkX graph.
    :return: The diameter of the graph.
    :rtype: Integer.
    """
    pass


def treewidth(graph):
    """The treewidth of the graph.

    :param graph: The graph.
    :type graph: NetworkX graph.
    :return: The treewidth of the graph.
    :rtype: Integer.
    """
    pass


def components(graph):
    """The number of connected components in the graph.

    :param graph: The graph.
    :type graph: NetworkX graph.
    :return: The number of connected components.
    :rtype: Integer.
    """
    pass


class Statistics:
    pass
