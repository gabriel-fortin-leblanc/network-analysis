import random

import networkx as nx
import numpy as np
import pytest

from networkanalysis.ergm.simulate import *
from networkanalysis.statistics import *


class TestSimulate:
    decay = 0.5
    k = 2
    ngraphs0 = 10
    ngraphs1 = 1000
    param0 = np.array([-1, 0.2, 0.5])
    param1 = np.array([-1, 1])
    param2 = np.array([1000, 0])
    stats_comp0 = StatsComp([NEdges(), GWD(decay), KStars(k)])
    stats_comp1 = StatsComp([NEdges(), Mutuals()])
    path = nx.path_graph(4)
    simple = nx.random_graphs.fast_gnp_random_graph(5, 0.5)
    complete = nx.complete_graph(4)
    n = 50
    burnin0 = 1000
    burnin1 = 0
    thin0 = 3
    thin1 = 1

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, n",
        [
            (ngraphs0, param0, stats_comp0, n),
            (ngraphs1, param1, stats_comp1, n),
        ],
    )
    def test_simulate_len_return(
        self, ngraphs, param, stats_comp, n, benchmark
    ):
        graphs = benchmark(
            simulate,
            ngraphs,
            param,
            stats_comp,
            n,
        )
        assert len(graphs) == ngraphs

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, n",
        [
            (ngraphs0, param0, stats_comp0, n),
            (ngraphs1, param1, stats_comp1, n),
        ],
    )
    def test_simulate_graphs_copy(
        self, ngraphs, param, stats_comp, n, benchmark
    ):
        graphs = benchmark(
            simulate,
            ngraphs,
            param,
            stats_comp,
            n,
        )
        assert graphs[0] != graphs[-1]

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, n, burnin, thin",
        [
            (ngraphs0, param0, stats_comp0, n, burnin0, thin0),
            (ngraphs1, param1, stats_comp1, n, burnin1, thin1),
        ],
    )
    def test_simulate_len_return_burnin_thin(
        self, ngraphs, param, stats_comp, n, burnin, thin, benchmark
    ):
        graphs = benchmark(
            simulate, ngraphs, param, stats_comp, n, burnin, thin
        )
        assert len(graphs) == ngraphs

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, graph_init",
        [
            (ngraphs0, param0, stats_comp0, path),
            (ngraphs1, param1, stats_comp1, simple),
        ],
    )
    def test_simulate_input_graphs(
        self, ngraphs, param, stats_comp, graph_init, benchmark
    ):
        graphs = benchmark(
            simulate,
            ngraphs,
            param,
            stats_comp,
            graph_init,
        )
        assert len(graphs) == ngraphs
        assert graphs[0] != graphs[-1]

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, n",
        [
            (ngraphs0, param0, stats_comp0, n),
            (ngraphs1, param1, stats_comp1, n),
        ],
    )
    def test_simulate_summary(self, ngraphs, param, stats_comp, n, benchmark):
        graphs, summary = benchmark(
            simulate, ngraphs, param, stats_comp, n, summary=True
        )
        assert summary is not None
        assert type(summary["rate"]) is float
        assert 0 <= summary["rate"] and summary["rate"] <= 1

    @pytest.mark.parametrize(
        "ngraphs, param, stats_comp, n, w",
        [
            (ngraphs1, param2, stats_comp1, complete, 1),
        ],
    )
    def test_simulate_degenerate_warning(
        self, ngraphs, param, stats_comp, n, w
    ):
        with pytest.warns(RuntimeWarning):
            graphs = simulate(
                ngraphs,
                param,
                stats_comp,
                n,
                warn=w,
            )
