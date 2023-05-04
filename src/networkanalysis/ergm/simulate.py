"""This module is used to simulate graphs from a exponential random graph
models.
"""
import random

import networkx as nx
import numpy as np


def simulate(
    ngraphs,
    param,
    stats_comp,
    init,
    burnin=None,
):
    """Simulate ngraphs graphs with respect to the param and the sufficient
    statistics.

    :param ngraphs: The number of graphs to simulate.
    :type ngraphs: Integer.
    :param param: The parameter of the model.
    :type param: Numpy array.
    :param stats_comp: The sufficient statistics computer.
    :type stats_comp: A callable object.
    :param init: The initial graph to start the chain, or the number of nodes.
    :type init: NetworkX graph., optional
    :param burnin: The number of graphs to burn. If none is given, then no
    graphs will be burned., defaults to None
    :type burnin: _type_, optional
    """
    if type(init) is int:
        peek = nx.random_graphs.binomial_graph(init, 0.5)
    else:
        peek = init.copy()
    peek_stats = stats_comp(peek)

    if burnin is None:
        burnin = 0

    nodes = list(peek.nodes())

    # Burnin phase
    for _ in range(burnin):
        peek, peek_stats = _next_state(
            peek, peek_stats, stats_comp, param, nodes
        )

    graphs = list()
    for _ in range(ngraphs):
        peek, peek_stats = _next_state(
            peek, peek_stats, stats_comp, param, nodes
        )
        graphs.append(peek.copy())

    return graphs


def _next_state(peek, peek_stats, stats_comp, param, nodes):
    u, v = random.sample(nodes, 2)
    in_graph = peek.has_edge(u, v)

    if in_graph:
        peek.remove_edge(u, v)
    else:
        peek.add_edge(u, v)
    temp_stats = stats_comp(peek)

    lunif = np.log(random.random())
    if lunif < param @ (temp_stats - peek_stats):  # Accept the state
        peek_stats = temp_stats
    else:  # Refuse the state
        if in_graph:
            peek.add_edge(u, v)
        else:
            peek.remove_edge(u, v)
    return peek, peek_stats
