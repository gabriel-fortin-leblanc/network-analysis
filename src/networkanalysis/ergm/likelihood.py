"""This module contains functions that compute the maximum likelihood estimator
or other version of it, like the maximum pseudolikelihood estimator."""
import itertools

import numpy as np
from scipy import optimize, spatial
from scipy.spatial._qhull import QhullError
from sklearn.linear_model import LogisticRegression

from networkanalysis.ergm.simulate import simulate

__all__ = ["mpl", "ml", "apl", "pl"]


def mpl(graph, statscomp, return_statscomp=False):
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
    X = np.array(X)
    y = np.array(y)

    # Fit the logistic regression.
    params = LogisticRegression(penalty=None).fit(X, y).coef_

    # Return phase
    if return_statscomp:
        return params.squeeze(), statscomp
    else:
        return params.squeeze()


def ml(
    graph,
    statscomp,
    init=None,
    ngraphs=500,
    burnin=500,
    thin=5,
    bound=1e-1,
    tol=1e-5,
    maxiter=50,
    return_statscomp=False,
):
    stats = statscomp(graph)
    param = mpl(graph, statscomp) if init is None else init

    it_in = 0
    it = 0
    while it_in < 2 and it < maxiter:
        it += 1

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
            sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
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
        sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
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

    return param, statscomp if return_statscomp else param


def apl(
    mple,
    mle,
    statscomp,
    ngraphs=500,
    burnin=500,
    thin=5,
):
    # Estimate the variance of sufficient statistics from the distribution
    # with the MPLE.
    sim_graphs, statscomp = simulate(
        ngraphs, mple, statscomp, burnin, thin, return_statscomp=True
    )
    sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
    mple_var = np.var(sim_stats, axis=0)

    # Estimate the variance of sufficient statistics from the distribution
    # with the MLE.
    sim_graphs, statscomp = simulate(
        ngraphs, mle, statscomp, burnin, thin, return_statscomp=True
    )
    sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
    mle_var = np.var(sim_stats, axis=0)

    #

    W = np.linalg.inv(
        np.linalg.cholesky(np.diag(-mple_var)).T
    ) @ np.linalg.inv(np.diag(-mle_var))


def ml(
    graph,
    statscomp,
    init=None,
    ngraphs=500,
    burnin=500,
    thin=5,
    bound=1e-1,
    tol=1e-5,
    maxiter=50,
    return_statscomp=False,
):
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
            sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
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
        sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
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


def pl(graph, param, statscomp, return_statscomp=False):
    stats = statscomp(graph)
    logmarglik = 0
    for u, v in itertools.combinations(graph.nodes(), 2):
        if graph.has_edge(u, v):  # Bit at 1
            graph.remove_edge(u, v)
            tempstats = statscomp(graph)
            graph.add_edge(u, v)
            diff = stats - tempstats
            diffparam = diff @ param
            logmarglik += diffparam - np.log(1 + np.exp(diffparam))
        else:  # Bit at 0
            graph.add_edge(u, v)
            tempstats = statscomp(graph)
            graph.remove_edge(u, v)
            diff = tempstats - stats
            logmarglik -= np.log(1 + np.exp(diff @ param))

    if return_statscomp:
        return np.exp(logmarglik), statscomp
    else:
        return np.exp(logmarglik)


def apl(
    mple,
    mle,
    statscomp,
    graph,
    ngraphs=500,
    burnin=500,
    thin=5,
    return_statscomp=False,
):
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
    sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
    mple_var = np.cov(sim_stats, rowvar=False)
    sim_graphs, statscomp = simulate(
        ngraphs,
        mle,
        statscomp,
        graph,
        burnin=burnin,
        thin=thin,
        return_statscomp=True,
    )
    sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
    mle_var = np.cov(sim_stats, rowvar=False)

    # Compute the curvature adjustment matrix.
    M = np.linalg.cholesky(mple_var).T
    N = np.linalg.cholesky(mle_var).T
    W = np.linalg.inv(M) @ N

    # Define the function that match-makes the moments of the pseudolikelihood
    # and the likelihood.
    def g(param):
        return mple + W @ (param - mle)

    # Estimate the normalizing constant of the likelihood for the MLE.
    temp = np.arange(0, 1, 0.01)
    ratios = np.empty((len(temp) - 1,))
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
        sim_stats = np.array([statscomp(sample) for sample in sim_graphs])
        ratios[i] = np.exp(sim_stats @ ((temp[i + 1] - temp[i]) * mle)).mean()
    n = graph.number_of_nodes()
    normconst = ratios.prod() * (2 ** (n * (n - 1) / 2))

    # Compute the magnitude adjustment constant.
    stats = statscomp(graph)
    pl_mple, statscomp = pl(graph, mple, statscomp, return_statscomp=True)
    C = np.exp(stats @ mle) / normconst / pl_mple

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
        stats @ (param - param0) - s_max - np.log(np.exp(s - s_max).mean())
    )


def _minus_gradient_log_ratio_likelihoods(param, param0, stats, sim_stats):
    s = sim_stats @ (param - param0)
    s_max = s.max(0)
    w = np.exp(s - s_max)
    return -(w[:, np.newaxis] * (stats - sim_stats) / w.sum()).sum(0)
