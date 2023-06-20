"""This module contains classes and functions to compute statistics on
networks.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, List, Union

import networkx as nx
import numpy as np
from scipy.special import comb

__all__ = [
    "NEdges",
    "GWD",
    "GWESP",
    "KStars",
    "InKStars",
    "OutKStars",
    "Mutuals",
    "gwd",
    "gwesp",
    "kstars",
    "in_kstars",
    "out_kstars",
    "mutuals",
    "stats_transform",
    "StatsComp",
    "CachedStatsComp",
]


def gwd(graph: nx.Graph, decay: float) -> float:
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


def gwesp(graph: nx.Graph, decay: float) -> float:
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


def kstars(graph: nx.Graph, k: int) -> int:
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


def in_kstars(graph: nx.DiGraph, k: int) -> int:
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


def out_kstars(graph: nx.DiGraph, k: int) -> int:
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


def mutuals(graph: Union[nx.Graph, nx.DiGraph]) -> int:
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
    adj = nx.to_numpy_array(graph)
    upper_diag_idx = np.triu_indices(adj.shape[0], 1)
    count_edges = adj[upper_diag_idx] + adj.T[upper_diag_idx]
    if not nx.is_directed(graph):
        count_edges /= 2
    return (count_edges > 1).sum()


class NEdges:
    """Dummy callable object that mimics number_of_edges of NetworkX."""

    def __call__(self, graph: Union[nx.Graph, nx.DiGraph]) -> int:
        return graph.number_of_edges()


class GWD:
    """Dummy callable object that mimics gwd."""

    def __init__(self, decay: float):
        self._decay = decay

    def __call__(self, graph: nx.Graph):
        return gwd(graph, self._decay)


class GWESP:
    """Dummy callable object that mimics gwesp."""

    def __init__(self, decay: float):
        self._decay = decay

    def __call__(self, graph: nx.Graph):
        return gwesp(graph, self._decay)


class KStars:
    """Dummy callable object that mimics kstars."""

    def __init__(self, k: int):
        self._k = k

    def __call__(self, graph: nx.Graph):
        return kstars(graph, self._k)


class InKStars:
    """Dummy callable object that mimics in_kstars."""

    def __init__(self, k: int):
        self._k = k

    def __call__(self, graph: nx.DiGraph):
        return in_kstars(graph, self._k)


class OutKStars:
    """Dummy callable object that mimics out_kstars."""

    def __init__(self, k: int):
        self._k = k

    def __call__(self, graph: nx.DiGraph):
        return out_kstars(graph, self._k)


class Mutuals:
    """Dummy callable object that mimics mutuals."""

    def __call__(self, graph: Union[nx.Graph, nx.DiGraph]):
        return mutuals(graph)


def stats_transform(stats: List[Callable]) -> Callable:
    """Transform the list of statistics into one function that computes the
    vector of statistics from the list.

    :param stats: List of callable object that takes a NetworkX graph as
        argument.
    :type stats: List.
    """

    # Build the function that computes the vector of statistics from the list.
    def stats_comp(graph):
        graph_stats = np.empty(
            (len(stats)),
        )
        for i, stat in enumerate(stats):
            graph_stats[i] = stat(graph)
        return graph_stats

    return stats_comp


class StatsComp:
    """Callable object that computes a vector of statistics from a graph.
    StatsComp can be understand as "Statistics Computer".
    """

    def __init__(self, stats: Union[List[Callable], StatsComp]):
        """Initialize the StatsComp object.

        :param stats: List of callable object that takes a NetworkX graph as
            argument. It also accepts another StatsComp object and copy it.
        :type stats: List of callable object or StatsComp object.
        """
        if isinstance(stats, StatsComp):
            self._func = stats._func
            self._len = stats._len
        else:
            self._func = stats_transform(stats)
            self._len = len(stats)

    def __len__(self) -> int:
        return self._len

    def __call__(self, graph) -> np.ndarray:
        return self._func(graph)


class CachedStatsComp(StatsComp):
    """StatsComp that caches the computed statistics."""

    def __init__(
        self,
        stats: Union[List[Callable], StatsComp, CachedStatsComp],
        max_size: int = 10000,
    ):
        """Initialize the CachedStatsComp object.

        :param stats: List of callable object that takes a NetworkX graph as
            argument. It also accepts another StatsComp object or even a
            CachedStatsComp object and copy it.
        :type stats: List of callable object, StatsComp object or
            CachedStatsComp object.
        :param max_size: Maximum number of graphs to cache.
        :type max_size: Integer.
        """
        super().__init__(stats)

        self._cache = OrderedDict()
        if isinstance(stats, CachedStatsComp):
            self._cache.update(stats._cache)
        self._max_size = max_size

    def __call__(self, graph: nx.Graph) -> np.ndarray:
        h = nx.weisfeiler_lehman_graph_hash(graph)
        if h in self._cache:
            return self._cache[h]

        if len(self._cache) >= self._max_size:
            self._cache.popitem()
        stats = self._func(graph)
        self._cache[graph] = stats
        return stats
