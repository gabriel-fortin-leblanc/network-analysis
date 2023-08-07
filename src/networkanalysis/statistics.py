"""This module contains classes and functions to compute statistics on
networks.
"""
from __future__ import annotations

import collections

import networkx
import numpy
from scipy import special

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

# Aliases
SeqGraph2Stats = list[
    collections.abc.Callable[
        [networkx.Graph | networkx.DiGraph],
        float | int]
]
Graph2StatsArray = collections.abc.Callable[
    [networkx.Graph | networkx.DiGraph], 
    numpy.ndarray
]


def gwd(graph: networkx.Graph, decay: float) -> float:
    """Compute the geometrically weighted degree of the simple graph.

    :param graph: The graph.
    :type graph: ~networkx.Graph
    :param decay: The decay parameter.
    :type decay: float
    :return: The geometrically weighted degree.
    :rtype: float
    """
    degrees = numpy.array([d for _, d in graph.degree()])
    uniques, counts = numpy.unique(degrees, return_counts=True)
    weighted_degrees = (1 - (1 - numpy.exp(-decay)) ** uniques) * counts
    return numpy.exp(decay) * weighted_degrees.sum()


def gwesp(graph: networkx.Graph, decay: float) -> float:
    """Compute the geometrically weighted edgewise shared partners of the
    graph.

    :param graph: The graph.
    :type graph: ~networkx.Graph
    :param decay: The decay parameter.
    :type decay: float
    :return: The geometrically weighted edgewise shared partners.
    :rtype: float
    """
    adj_matrix = networkx.to_numpy_array(graph)
    n_common_neighbours = (adj_matrix @ adj_matrix) * adj_matrix
    upper_diag_idx = numpy.triu_indices(adj_matrix.shape[0], 1)
    uniques, counts = numpy.unique(
        n_common_neighbours[upper_diag_idx], return_counts=True
    )
    weighted_ew_shared_partners = (
        1 - (1 - numpy.exp(-decay)) ** uniques
    ) * counts
    return numpy.exp(decay) * weighted_ew_shared_partners.sum()


def kstars(graph: networkx.Graph, k: int) -> int:
    """Count the number of k-stars of the undirected graph.

    :param graph: The graph.
    :type graph: ~networkx.Graph
    :param k: The number of branch of a star.
    :type k: int
    :return: The number of k-stars the graph contains.
    :rtype: int
    """
    degrees = numpy.array([d for _, d in graph.degree()])
    return collections.comb(degrees, k).sum()


def in_kstars(graph: networkx.DiGraph, k: int) -> int:
    """Count the number of in k-stars of the directed graph. An in k-star in
    composed of arcs pointing towards its center.

    :param graph: The graph.
    :type graph: ~networkx.DiGraph
    :param k: The number of branch of a star.
    :type k: int
    :return: The number of in k-stars the graph contains.
    :rtype: int
    """
    degrees = numpy.array([d for _, d in graph.in_degree()])
    return special.comb(degrees, k).sum()


def out_kstars(graph: networkx.DiGraph, k: int) -> int:
    """Count the number of out k-stars of the directed graph. An out k-star in
    composed of arcs pointing towards its border.

    :param graph: The graph.
    :type graph: ~networkx.DiGraph
    :param k: The number of branch of a star.
    :type k: int
    :return: The number of out k-stars the graph contains.
    :rtype: int
    """
    degrees = numpy.array([d for _, d in graph.out_degree()])
    return special.comb(degrees, k).sum()


def mutuals(graph: networkx.Graph | networkx.DiGraph) -> int:
    """Count the number of pairs of nodes in the graph that has a mutual
    connection. In a undirected multigraph, two nodes have a mutual connection
    if there are at least two edges between them. In a directed graph, two arcs
    between two nodes must be of opposite direction for having a mutual
    connection.

    :param graph: The graph.
    :type graph: ~networkx.Graph | ~networkx.DiGraph
    :return: The number of mutual connections.
    :rtype: int
    """
    adj = networkx.to_numpy_array(graph)
    upper_diag_idx = numpy.triu_indices(adj.shape[0], 1)
    count_edges = adj[upper_diag_idx] + adj.T[upper_diag_idx]
    if not networkx.is_directed(graph):
        count_edges /= 2
    return (count_edges > 1).sum()


class NEdges:
    """Dummy callable object that mimics
    :py:meth:`~networkx.Graph.number_of_edges`.
    """

    def __call__(self, graph: networkx.Graph | networkx.DiGraph) -> int:
        return graph.number_of_edges()


class GWD:
    """Dummy callable object that mimics :py:func:`gwd`."""

    def __init__(self, decay: float):
        """Initializa a callable object that mimics :py:func:`gwd`.

        :param decay: The decay parameter.
        :type decay: float
        """
        self._decay = decay

    def __call__(self, graph: networkx.Graph) -> float:
        return gwd(graph, self._decay)


class GWESP:
    """Dummy callable object that mimics :py:func:`gwesp`."""

    def __init__(self, decay: float):
        """Initializa a callable object that mimics :py:func:`gwesp`.
        
        :param decay: The decay parameter.
        :type decay: float
        """
        self._decay = decay

    def __call__(self, graph: networkx.Graph) -> float:
        return gwesp(graph, self._decay)


class KStars:
    """Dummy callable object that mimics :py:func:`kstars`."""

    def __init__(self, k: int):
        """Initializa a callable object that mimics :py:func:`kstars`.

        :param k: The number of branch of a star.
        :type k: int
        """
        self._k = k

    def __call__(self, graph: networkx.Graph) -> int:
        return kstars(graph, self._k)


class InKStars:
    """Dummy callable object that mimics :py:func:`in_kstars`."""

    def __init__(self, k: int):
        """Initializa a callable object that mimics :py:func:`in_kstars`.

        :param k: The number of branch of a star.
        :type k: int
        """
        self._k = k

    def __call__(self, graph: networkx.DiGraph) -> int:
        return in_kstars(graph, self._k)


class OutKStars:
    """Dummy callable object that mimics :py:func:`out_kstars`."""

    def __init__(self, k: int):
        """Initializa a callable object that mimics :py:func:`out_kstars`.

        :param k: The number of branch of a star.
        :type k: int
        """
        self._k = k

    def __call__(self, graph: networkx.DiGraph) -> int:
        return out_kstars(graph, self._k)


class Mutuals:
    """Dummy callable object that mimics :py:func:`mutuals`."""

    def __call__(self, graph: networkx.Graph | networkx.DiGraph) -> int:
        return mutuals(graph)


def stats_transform(stats: SeqGraph2Stats) -> Graph2StatsArray:
    """Transform the list of statistics into one function that computes the
    vector of statistics from the list.

    :param stats: List of callable object that takes a NetworkX graph as
        argument.
    :type stats: SeqGraph2Stats
    :return: A callable object that takes a NetworkX graph as argument and
        returns a vector of statistics.
    :rtype: collections.abc.Callable[
        [networkx.Graph | networkx.DiGraph], 
        numpy.ndarray]
    """

    # Build the function that computes the vector of statistics from the list.
    def stats_comp(graph):
        graph_stats = numpy.empty(
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

    def __init__(self, stats: SeqGraph2Stats | StatsComp):
        """Initialize the StatsComp object.

        :param stats: List of callable object that takes a NetworkX graph as
            argument. It also accepts another StatsComp object and copy it.
        :type stats: list[
            ~collections.abc.Callable[
            [~networkx.Graph | ~networkx.DiGraph],
            float | int]
            ]
        """
        if isinstance(stats, StatsComp):
            self._func = stats._func
            self._len = stats._len
        else:
            self._func = stats_transform(stats)
            self._len = len(stats)

    def __len__(self) -> int:
        return self._len

    def __call__(self, graph) -> numpy.ndarray:
        return self._func(graph)


class CachedStatsComp(StatsComp):
    """:py:class:`StatsComp` that caches the computed statistics."""

    def __init__(
        self,
        stats: SeqGraph2Stats | StatsComp | CachedStatsComp,
        max_size: int = 10000,
    ):
        """Initialize the CachedStatsComp object.

        :param stats: List of callable object that takes a NetworkX graph as
            argument. It also accepts another StatsComp object or even a
            CachedStatsComp object and copy it.
        :type stats: list[
            collections.abc.Callable[
            [networkx.Graph | networkx.DiGraph],
            float | int]
            ] | StatsComp | CachedStatsComp
        :param max_size: Maximum number of graphs to cache.
        :type max_size: int
        """
        super().__init__(stats)

        self._cache = collections.OrderedDict()
        if isinstance(stats, CachedStatsComp):
            self._cache.update(stats._cache)
        self._max_size = max_size

    def __call__(self, graph: networkx.Graph) -> numpy.ndarray:
        h = networkx.weisfeiler_lehman_graph_hash(graph)
        if h in self._cache:
            return self._cache[h]

        if len(self._cache) >= self._max_size:
            self._cache.popitem()
        stats = self._func(graph)
        self._cache[graph] = stats
        return stats
