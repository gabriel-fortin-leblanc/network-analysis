"""This module is used to simulate graphs from a exponential random graph
models.
"""
import random
import warnings

import networkx as nx
import numpy as np


def simulate(
    ngraphs,
    param,
    stats_comp,
    init,
    burnin=None,
    thin=None,
    summary=False,
    warn=None,
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
    :type burnin: int, optional
    :param thin: The thinning factor. By default, None is given.
    :type thin: int, optional.
    :param summary: A flag for requesting to collect information about the
    chain such as the acceptance rate. By default, False is given.
    :type summary: Boolean.
    :param warn: If an integer passed, then a warning is thrown if the
    graphs are near-empty or near-complete for this number of interation.
    :type warn: An integer, optional.
    """
    if type(init) is int:
        peek = nx.random_graphs.binomial_graph(init, 0.5)
    else:
        peek = init.copy()
    peek_stats = stats_comp(peek)

    nodes = list(peek.nodes())
    current = (peek, peek_stats)

    if burnin is None:
        burnin = 0

    if thin is None:
        thin = 1

    if summary:
        naccepted = 0

    if warn is not None:
        ndeg = 0
        n = peek.number_of_nodes()
        ndyadslim = n * (n - 1) / 2 - 1

    # Burnin phase
    for _ in range(burnin):
        current = _next_state(
            current[0], current[1], stats_comp, param, nodes, summary
        )
        if summary:
            naccepted += current[2]
        if warn is not None:
            if (
                current[0].number_of_edges() <= 1
                or current[0].number_of_edges() >= ndyadslim
            ):
                ndeg += 1
            else:
                ndeg = 0
            if ndeg >= warn:
                warnings.warn(
                    message="The graph states are near-degenerated.",
                    category=RuntimeWarning,
                )

    graphs = list()
    for _ in range(ngraphs * thin):
        current = _next_state(
            current[0], current[1], stats_comp, param, nodes, summary
        )
        graphs.append(current[0].copy())
        if summary:
            naccepted += current[2]

        if warn is not None:
            print(current[0].number_of_edges())
            print(ndyadslim)
            if (
                current[0].number_of_edges() <= 1
                or current[0].number_of_edges() >= ndyadslim
            ):
                ndeg += 1
            else:
                ndeg = 0
            if ndeg >= warn:
                warnings.warn(
                    message="The graph states are near-degenerated.",
                    category=RuntimeWarning,
                )

    graphs = graphs[::thin]
    if summary:
        return graphs, {"rate": naccepted / (ngraphs * thin + burnin)}
    return graphs


def _next_state(peek, peek_stats, stats_comp, param, nodes, summary=False):
    u, v = random.sample(nodes, 2)
    in_graph = peek.has_edge(u, v)

    if in_graph:
        peek.remove_edge(u, v)
    else:
        peek.add_edge(u, v)
    temp_stats = stats_comp(peek)

    lunif = np.log(random.random())
    if lunif < param @ (temp_stats - peek_stats):  # Accept the state
        if summary:
            accept = 1
        peek_stats = temp_stats
    else:  # Refuse the state
        if summary:
            accept = 0
        if in_graph:
            peek.add_edge(u, v)
        else:
            peek.remove_edge(u, v)

    if summary:
        return peek, peek_stats, accept
    return peek, peek_stats
