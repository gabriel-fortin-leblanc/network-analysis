"""This module contains functions that compute the maximum likelihood estimator
or other version of it, like the maximum pseudolikelihood estimator."""
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = ["mple"]


def mple(graph, stats_comp):
    # Create the "data set".
    X = list()
    y = list()
    stat0 = stat1 = None
    for u, v in itertools.combinations(graph.nodes(), 2):
        if graph.has_edge(u, v):  # Bit at 1
            stat1 = stats_comp(graph)
            graph.remove_edge(u, v)
            stat0 = stats_comp(graph)
            graph.add_edge(u, v)
            y.append(1)
        else:  # Bit at 0
            stat0 = stats_comp(graph)
            graph.add_edge(u, v)
            stat1 = stats_comp(graph)
            graph.remove_edge(u, v)
            y.append(0)
        X.append(stat1 - stat0)
    X = np.array(X)
    y = np.array(y)
    params = (
        LogisticRegression(penalty=None, fit_intercept=False).fit(X, y).coef_
    )
    return params
