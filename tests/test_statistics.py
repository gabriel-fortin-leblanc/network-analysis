import networkx as nx
import numpy as np
import pytest
from pytest import approx

from networkanalysis.statistics import *

# Create undirected graphs.
path = nx.path_graph(4)
cycle = nx.cycle_graph(5)
complete = nx.complete_graph(4)
bi_complete = nx.complete_bipartite_graph(2, 3)
custom = nx.Graph()
custom.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5)])

# Create directed graphs.
dpath = nx.DiGraph()
dpath.add_edges_from([(0, 1), (1, 2), (3, 2)])
din_star = nx.DiGraph()
din_star.add_edges_from([(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
dout_star = nx.DiGraph()
dout_star.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
dcustom = nx.DiGraph()
dcustom.add_edges_from([(0, 1), (1, 2), (2, 3), (4, 1), (4, 3)])

# Create mutligraphs.
mcustom = nx.MultiGraph()
mcustom.add_edges_from(
    [(0, 1), (0, 1), (0, 1), (1, 2), (1, 3), (2, 3), (2, 3)]
)

# Create multi-directed graphs.
mdcustom = nx.MultiDiGraph()
mdcustom.add_edges_from([(0, 2), (0, 3), (1, 0), (1, 2), (2, 1)])


class TestGWD:
    # Decay factors
    decay0 = 0.5
    decay1 = 2

    @pytest.mark.parametrize(
        "graph, decay, expected",
        [
            (path, decay0, approx(4.786938)),
            (path, decay1, approx(5.729329)),
            (cycle, decay0, approx(6.967346)),
            (cycle, decay1, approx(9.323323)),
            (complete, decay0, approx(6.193149)),
            (complete, decay1, approx(10.449239)),
            (bi_complete, decay0, approx(7.276982)),
            (bi_complete, decay1, approx(10.818613)),
            (custom, decay0, approx(8.337899)),
            (custom, decay1, approx(12.465076)),
        ],
    )
    def test_gwd(self, graph, decay, expected, benchmark):
        assert benchmark(gwd, graph, decay) == expected


class TestGWESP:
    # Decay factors
    decay0 = 0.5
    decay1 = 2

    @pytest.mark.parametrize(
        "graph, decay, expected",
        [
            (path, decay0, approx(0.0)),
            (path, decay1, approx(0.0)),
            (cycle, decay0, approx(0.0)),
            (cycle, decay1, approx(0.0)),
            (complete, decay0, approx(8.360816)),
            (complete, decay1, approx(11.187988)),
            (bi_complete, decay0, approx(0.0)),
            (bi_complete, decay1, approx(0.0)),
            (custom, decay0, approx(6.0)),
            (custom, decay1, approx(6.0)),
        ],
    )
    def test_gwesp(self, graph, decay, expected, benchmark):
        assert benchmark(gwesp, graph, decay) == expected


class TestKStars:
    # K factors
    k0 = 2
    k1 = 3

    @pytest.mark.parametrize(
        "graph, k, expected",
        [
            (path, k0, 2),
            (path, k1, 0),
            (cycle, k0, 5),
            (cycle, k1, 0),
            (complete, k0, 12),
            (complete, k1, 4),
            (bi_complete, k0, 9),
            (bi_complete, k1, 2),
            (custom, k0, 12),
            (custom, k1, 5),
        ],
    )
    def test_kstars(self, graph, k, expected, benchmark):
        assert benchmark(kstars, graph, k) == expected


class TestKInStars:
    # K factors
    k0 = 2
    k1 = 3

    @pytest.mark.parametrize(
        "graph, k, expected",
        [
            (dpath, k0, 1),
            (dpath, k1, 0),
            (din_star, k0, 10),
            (din_star, k1, 10),
            (dout_star, k0, 0),
            (dout_star, k1, 0),
        ],
    )
    def test_in_kstars(self, graph, k, expected, benchmark):
        assert benchmark(in_kstars, graph, k) == expected


class TestKOutStars:
    # K factors
    k0 = 2
    k1 = 3

    @pytest.mark.parametrize(
        "graph, k, expected",
        [
            (dpath, k0, 0),
            (dpath, k1, 0),
            (din_star, k0, 0),
            (din_star, k1, 0),
            (dout_star, k0, 10),
            (dout_star, k1, 10),
        ],
    )
    def test_out_kstars(self, graph, k, expected, benchmark):
        assert benchmark(out_kstars, graph, k) == expected


class TestMutuals:
    @pytest.mark.parametrize("graph, expected", [(mcustom, 2), (mdcustom, 1)])
    def test_mutuals(self, graph, expected, benchmark):
        assert benchmark(mutuals, graph) == expected


class TestStatsTransform:
    # Factors
    decay = 0.5
    k0 = 2
    k1 = 3

    @pytest.mark.parametrize(
        "stats_comp, graph, expected",
        [
            (
                stats_transform([NEdges(), GWD(decay), KStars(k0)]),
                path,
                np.array([3, 4.786938, 2]),
            ),
            (
                stats_transform([NEdges(), GWD(decay), KStars(k0)]),
                custom,
                np.array([7, 8.337899, 12]),
            ),
            (
                stats_transform([InKStars(k0), OutKStars(k1)]),
                dpath,
                np.array([1, 0]),
            ),
            (
                stats_transform([InKStars(k0), OutKStars(k1)]),
                din_star,
                np.array([10, 0]),
            ),
        ],
    )
    def test_stats_transform(self, stats_comp, graph, expected):
        np.testing.assert_almost_equal(stats_comp(graph), expected, 5)


class TestCachedStatsComp:
    # Factors
    decay = 0.5
    k0 = 2
    k1 = 3

    @pytest.mark.parametrize(
        "stats_comp, graph, expected",
        [
            (
                CachedStatsComp([NEdges(), GWD(decay), KStars(k0)], 2),
                path,
                np.array([3, 4.786938, 2]),
            ),
            (
                CachedStatsComp([NEdges(), GWD(decay), KStars(k0)], 2),
                custom,
                np.array([7, 8.337899, 12]),
            ),
            (
                CachedStatsComp([InKStars(k0), OutKStars(k1)], 2),
                dpath,
                np.array([1, 0]),
            ),
            (
                CachedStatsComp([InKStars(k0), OutKStars(k1)], 2),
                din_star,
                np.array([10, 0]),
            ),
        ],
    )
    def test_stats_transform(self, stats_comp, graph, expected):
        stats_comp(graph)
        stats_comp(
            nx.gn_graph(10)
            if nx.is_directed(graph)
            else nx.fast_gnp_random_graph(10, 0.5)
        )
        np.testing.assert_almost_equal(stats_comp(graph), expected, 5)
