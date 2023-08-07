"""This module contains functions that compute the maximum likelihood estimator
or other version of it, like the maximum pseudolikelihood estimator."""
from __future__ import annotations

import itertools
from collections import abc

import networkx
import numpy
from scipy import optimize, spatial
from scipy.spatial._qhull import QhullError
from sklearn.linear_model import LogisticRegression

from ..statistics import CachedStatsComp, StatsComp
from .simulate import simulate

__all__ = ["mpl", "ml", "apl", "pl"]


def mpl(
    graph: networkx.Graph,
    statscomp: StatsComp | CachedStatsComp,
    return_statscomp: bool = False,
) -> numpy.ndarray:
    """Compute the maximum pseudolikelihood estimator.

    :param graph: The observed graph.
    :type graph: ~networkx.Graph
    :param statscomp: The statistics computer.
    :type statscomp: StatsComp | CachedStatsComp
    :param return_statscomp: Whether to return the function that computes the
        sufficient statistics. Defaults to `False`.
    :type return_statscomp: bool, optional
    :return:
        - The maximum pseudolikelihood estimator.
        - The statistics computer if `return_statscomp` is `True`.
    :rtype: tuple[~numpy.ndarray, StatsComp | CachedStatsComp]
    """
    # Create the "data set".
    X = list()
    y = list()
    stat0 = stat1 = None
    for u, v in itertools.combinations(graph.nodes(), 2):
        if graph.has_edge(u, v):  # Bit at 1
            stat1 = statscomp(graph)
            graph.remove_edge(u, v)
            stat0 = statscomp(graph)
            graph.add_edge(u, v)
            y.append(1)
        else:  # Bit at 0
            stat0 = statscomp(graph)
            graph.add_edge(u, v)
            stat1 = statscomp(graph)
            graph.remove_edge(u, v)
            y.append(0)
        X.append(stat1 - stat0)
    X = numpy.array(X)
    y = numpy.array(y)

    # Fit the logistic regression.
    params = LogisticRegression(penalty=None).fit(X, y).coef_

    # Return phase
    if return_statscomp:
        return params.squeeze(), statscomp
    else:
        return params.squeeze()


def ml(
    graph: networkx.Graph,
    statscomp: StatsComp | CachedStatsComp,
    init: numpy.ndarray = None,
    ngraphs: int = 500,
    burnin: int = 500,
    thin: int = 5,
    bound: float = 1e-1,
    tol: float = 1e-5,
    maxiter: int = 50,
    return_statscomp: bool = False,
) -> numpy.ndarray:
    """Compute the maximum likelihood estimator.

    :param graph: The graph to compute the estimator.
    :type graph: ~networkx.Graph
    :param statscomp: The function that computes the sufficient statistics.
    :type statscomp: StatsComp | CachedStatsComp
    :param init: The initial parameter to start the optimization. If None, the
        maximum pseudolikelihood estimator is used. Defaults to None.
    :type init: ~numpy.ndarray, optional
    :param ngraphs: The number of graphs to use when using simulate. Defaults
        to 500.
    :type ngraphs: int, optional
    :param burnin: The number of graphs to discard when using simulate.
        Defaults to `500`.
    :type burnin: int, optional
    :param thin: The thinning parameter when using simulate. Defaults to `5`.
    :type thin: int, optional
    :param bound: The bound for the optimization. Defaults to `1e-1`.
    :type bound: float, optional
    :param tol: The tolerance for the optimization. Defaults to `1e-5`.
    :type tol: float, optional
    :param maxiter: The maximum number of iterations for the current two-phases
        algorithm. Defaults to `50`.
    :type maxiter: int, optional
    :param return_statscomp: Whether to return the function that computes the
        sufficient statistics. Defaults to `False`.
    :type return_statscomp: bool, optional
    :return: The maximum likelihood estimator.
    :rtype: tuple[~numpy.ndarray, StatsComp | CachedStatsComp]
    """
    stats = statscomp(graph)
    if init is None:
        param, statscomp = mpl(graph, statscomp, return_statscomp=True)
    else:
        param = init

    it_in = 0
    it = 0
    while it_in < 2 and it < maxiter:
        it += 1

        # Find the convex hull of the sufficient statistics of the simulated
        # graphs.
        hull = None
        factor = 1
        while hull is None:
            sim_graphs, statscomp = simulate(
                ngraphs,
                param,
                statscomp,
                graph,
                burnin,
                thin,
                return_statscomp=True,
            )
            sim_stats = numpy.array(
                [statscomp(sample) for sample in sim_graphs]
            )
            sim_stats_mean = sim_stats.mean(0)
            try:
                hull = spatial.Delaunay(sim_stats)
            except QhullError:
                if factor == 8:
                    return (
                        "The algorithm did not converge. "
                        "The sufficient statistics of the generated "
                        "graphs are always affinely dependent."
                    )
                factor *= 2

        # Find greatest gamma which we have the pseudo-observation in the
        # convex hull.
        inf_gamma = 0
        sup_gamma = 1
        in_hull = False
        while sup_gamma - inf_gamma > tol or not in_hull:
            gamma = (sup_gamma + inf_gamma) / 2
            pseudostats = gamma * stats + (1 - gamma) * sim_stats_mean
            in_hull = hull.find_simplex(pseudostats) > 0
            if in_hull:
                inf_gamma = gamma
            else:
                sup_gamma = gamma
        gamma = (sup_gamma + inf_gamma) / 2
        pseudostats = gamma * stats + (1 - gamma) * sim_stats_mean

        # Optimization of the approximation of the log-likelihood.
        param = optimize.minimize(
            _minus_log_ratio_likelihoods,
            param,
            args=(param, pseudostats, sim_stats),
            jac=_minus_gradient_log_ratio_likelihoods,
            bounds=optimize.Bounds(param - bound, param + bound),
            tol=tol,
        ).x

        if gamma > 1 - tol:
            it_in += 1
        else:
            it_in = 0

    # Last iteration.
    in_hull = False
    while not in_hull:
        sim_graphs, statscomp = simulate(
            ngraphs,
            param,
            statscomp,
            graph,
            burnin,
            thin,
            return_statscomp=True,
        )
        sim_stats = numpy.array([statscomp(sample) for sample in sim_graphs])
        try:
            hull = spatial.Delaunay(sim_stats)
            in_hull = hull.find_simplex(stats) > 0
        except QhullError:
            pass

    param = optimize.minimize(
        _minus_log_ratio_likelihoods,
        param,
        args=(param, stats, sim_stats),
        jac=_minus_gradient_log_ratio_likelihoods,
        tol=tol,
    ).x

    # Return phase
    if return_statscomp:
        return param, statscomp
    else:
        return param


def pl(
    graph: networkx.Graph,
    param: numpy.ndarray,
    statscomp: StatsComp | CachedStatsComp,
    return_statscomp: bool = False,
) -> numpy.ndarray | tuple[numpy.ndarray, StatsComp | CachedStatsComp]:
    """Compute the pseudolikelihood of a graph.

    :param graph: The graph.
    :type graph: ~networkx.Graph
    :param param: The parameter.
    :type param: ~numpy.ndarray
    :param statscomp: The function that computes the sufficient statistics.
    :type statscomp: StatsComp | CachedStatsComp
    :param return_statscomp: Whether to return the function that computes the
        sufficient statistics. Defaults to `False`.
    :type return_statscomp: bool, optional
    :return:
        - The pseudolikelihood of the graph.
        - The statistics computer if `return_statscomp` is `True`.
    :rtype: tuple[~numpy.ndarray, StatsComp | CachedStatsComp]
    """
    stats = statscomp(graph)
    logmarglik = 0
    for u, v in itertools.combinations(graph.nodes(), 2):
        if graph.has_edge(u, v):  # Bit at 1
            graph.remove_edge(u, v)
            tempstats = statscomp(graph)
            graph.add_edge(u, v)
            diff = stats - tempstats
            diffparam = diff @ param
            logmarglik += diffparam - numpy.log(1 + numpy.exp(diffparam))
        else:  # Bit at 0
            graph.add_edge(u, v)
            tempstats = statscomp(graph)
            graph.remove_edge(u, v)
            diff = tempstats - stats
            logmarglik -= numpy.log(1 + numpy.exp(diff @ param))

    if return_statscomp:
        return numpy.exp(logmarglik), statscomp
    else:
        return numpy.exp(logmarglik)


def apl(
    mple: numpy.ndarray,
    mle: numpy.ndarray,
    statscomp: callable,
    graph: networkx.Graph,
    ngraphs: int = 500,
    burnin: int = 500,
    thin: int = 5,
    return_statscomp=False,
) -> (
    abc.Callable[[numpy.ndarray], float]
    | tuple[abc.Callable[[numpy.ndarray], float], StatsComp | CachedStatsComp]
):
    """Compute the adjusted pseudolikelihood function of a graph.

    :param mple: The maximum pseudolikelihood estimator.
    :type mple: ~numpy.ndarray
    :param mle: The maximum likelihood estimator.
    :type mle: ~numpy.ndarray
    :param statscomp: The function that computes the sufficient statistics.
    :type statscomp: StatsComp | CachedStatsComp
    :param graph: The observed graph.
    :type graph: ~networkx.Graph
    :param ngraphs: The number of graphs to simulate. Defaults to `500`.
    :type ngraphs: int, optional
    :param burnin: The number of iterations to discard. Defaults to `500`.
    :type burnin: int, optional
    :param thin: The thinning parameter. Defaults to `5`.
    :type thin: int, optional
    :param return_statscomp: Whether to return the function that computes the
        sufficient statistics. Defaults to `False`.
    :type return_statscomp: bool, optional
    :return:
        - The adjusted pseudolikelihood function with respect to the observed
            graph.
        - The statistics computer if `return_statscomp` is `True`.
    :rtype: tuple[~collections.abc.Callable[[numpy.ndarray], float],
        StatsComp | CachedStatsComp]
    """
    # Compute the variance of the sufficient statistics for the MLE and the
    # MPLE.
    sim_graphs, statscomp = simulate(
        ngraphs,
        mple,
        statscomp,
        graph,
        burnin=burnin,
        thin=thin,
        return_statscomp=True,
    )
    sim_stats = numpy.array([statscomp(sample) for sample in sim_graphs])
    mple_var = numpy.cov(sim_stats, rowvar=False)
    sim_graphs, statscomp = simulate(
        ngraphs,
        mle,
        statscomp,
        graph,
        burnin=burnin,
        thin=thin,
        return_statscomp=True,
    )
    sim_stats = numpy.array([statscomp(sample) for sample in sim_graphs])
    mle_var = numpy.cov(sim_stats, rowvar=False)

    # Compute the curvature adjustment matrix.
    M = numpy.linalg.cholesky(mple_var).T
    N = numpy.linalg.cholesky(mle_var).T
    W = numpy.linalg.inv(M) @ N

    # Define the function that match-makes the moments of the pseudolikelihood
    # and the likelihood.
    def g(param):
        return mple + W @ (param - mle)

    # Estimate the normalizing constant of the likelihood for the MLE.
    temp = numpy.arange(0, 1, 0.01)
    ratios = numpy.empty((len(temp) - 1,))
    for i in range(len(ratios)):
        sim_graphs, statscomp = simulate(
            ngraphs,
            temp[i] * mle,
            statscomp,
            graph,
            burnin=burnin,
            thin=thin,
            return_statscomp=True,
        )
        sim_stats = numpy.array([statscomp(sample) for sample in sim_graphs])
        ratios[i] = numpy.exp(
            sim_stats @ ((temp[i + 1] - temp[i]) * mle)
        ).mean()
    n = graph.number_of_nodes()
    normconst = ratios.prod() * (2 ** (n * (n - 1) / 2))

    # Compute the magnitude adjustment constant.
    stats = statscomp(graph)
    pl_mple, statscomp = pl(graph, mple, statscomp, return_statscomp=True)
    C = numpy.exp(stats @ mle) / normconst / pl_mple

    # Compute the adjusted pseudolikelihood.
    def f(gr, pa):
        return C * pl(gr, g(pa), statscomp)

    if return_statscomp:
        return f, statscomp
    else:
        return f


def _minus_log_ratio_likelihoods(param, param0, stats, sim_stats):
    s = sim_stats @ (param - param0)
    s_max = s.max(0)
    return -(
        stats @ (param - param0)
        - s_max
        - numpy.log(numpy.exp(s - s_max).mean())
    )


def _minus_gradient_log_ratio_likelihoods(param, param0, stats, sim_stats):
    s = sim_stats @ (param - param0)
    s_max = s.max(0)
    w = numpy.exp(s - s_max)
    return -(w[:, numpy.newaxis] * (stats - sim_stats) / w.sum()).sum(0)
