import random

import networkx as nx
import numpy as np
import pytest

from networkanalysis.ergm.likelihood import *
from networkanalysis.statistics import *

# Create undirected graphs.
path = nx.path_graph(4)
cycle = nx.cycle_graph(5)
complete = nx.complete_graph(4)
bi_complete = nx.complete_bipartite_graph(2, 3)
custom = nx.Graph()
custom.add_edges_from(
    [
        (0, 1),
        (0, 2),
        (0, 5),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
    ]
)
random.seed(1234)
np.random.seed(1234)
rg0 = nx.erdos_renyi_graph(20, 0.4)

# Create models.
decay = 0.5
k = 2
stats_comp0 = StatsComp([NEdges(), GWD(decay), KStars(k)])
stats_comp1 = StatsComp([NEdges(), Mutuals()])
param0 = np.array([-2, 0.5, 0.5])
param1 = np.array([-2, 0.5])


class TestMPL:
    @pytest.mark.parametrize(
        "graph, stats_comp",
        [
            (path, stats_comp0),
            (path, stats_comp1),
            (cycle, stats_comp0),
            (cycle, stats_comp1),
        ],
    )
    def test_mpl(self, graph, stats_comp, benchmark):
        res = benchmark(mpl, graph, stats_comp)
        assert type(res) is np.ndarray
        assert res.shape == (len(stats_comp),)


class TestPL:
    @pytest.mark.parametrize(
        "graph, param, stats_comp",
        [
            (path, param0, stats_comp0),
            (path, param1, stats_comp1),
            (cycle, param0, stats_comp0),
            (cycle, param1, stats_comp1),
        ],
    )
    def test_pl(self, graph, param, stats_comp, benchmark):
        res = benchmark(pl, graph, param, stats_comp)
        assert 0 <= res <= 1


class TestML:
    @pytest.mark.parametrize(
        "graph, stats_comp, init",
        [
            (rg0, stats_comp0, mpl(rg0, stats_comp0)),
        ],
    )
    def test_ml(self, graph, stats_comp, init, benchmark):
        res = benchmark(ml, graph, stats_comp, init)
        assert type(res) is np.ndarray
        assert res.shape == (len(stats_comp),)


class TestAPL:
    rg0_mple = mpl(rg0, stats_comp0)
    rg0_mle = ml(rg0, stats_comp0, rg0_mple)

    @pytest.mark.parametrize(
        "mple, mle, stats_comp, graph",
        [
            (rg0_mple, rg0_mle, stats_comp0, rg0),
        ],
    )
    def test_apl(self, mple, mle, stats_comp, graph, benchmark):
        f = benchmark(apl, mple, mle, stats_comp, graph)
        assert callable(f)
        assert 0 <= f(graph, mple) <= 1
        assert 0 <= f(graph, mle) <= 1
