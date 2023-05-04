import networkx as nx
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


def test_gwd():
    decay0 = 0.5
    decay1 = 2

    assert gwd(path, decay0) == approx(4.786938)
    assert gwd(path, decay1) == approx(5.729329)
    assert gwd(cycle, decay0) == approx(6.967346)
    assert gwd(cycle, decay1) == approx(9.323323)
    assert gwd(complete, decay0) == approx(6.193149)
    assert gwd(complete, decay1) == approx(10.449239)
    assert gwd(bi_complete, decay0) == approx(7.276982)
    assert gwd(bi_complete, decay1) == approx(10.818613)
    assert gwd(custom, decay0) == approx(8.337899)
    assert gwd(custom, decay1) == approx(12.465076)


def test_gwesp():
    decay0 = 0.5
    decay1 = 2

    assert gwesp(path, decay0) == approx(0.0)
    assert gwesp(path, decay1) == approx(0.0)
    assert gwesp(cycle, decay0) == approx(0.0)
    assert gwesp(cycle, decay1) == approx(0.0)
    assert gwesp(complete, decay0) == approx(8.360816)
    assert gwesp(complete, decay1) == approx(11.187988)
    assert gwesp(bi_complete, decay0) == approx(0.0)
    assert gwesp(bi_complete, decay1) == approx(0.0)
    assert gwesp(custom, decay0) == approx(6.0)
    assert gwesp(custom, decay1) == approx(6.0)


def test_kstars():
    k0 = 2
    k1 = 3

    assert kstars(path, k0) == 2
    assert kstars(path, k1) == 0
    assert kstars(cycle, k0) == 5
    assert kstars(cycle, k1) == 0
    assert kstars(complete, k0) == 12
    assert kstars(complete, k1) == 4
    assert kstars(bi_complete, k0) == 9
    assert kstars(bi_complete, k1) == 2
    assert kstars(custom, k0) == 12
    assert kstars(custom, k1) == 5


def test_in_kstars():
    k0 = 2
    k1 = 3

    assert in_kstars(dpath, k0) == 1
    assert in_kstars(dpath, k1) == 0
    assert in_kstars(din_star, k0) == 10
    assert in_kstars(din_star, k1) == 10
    assert in_kstars(dout_star, k0) == 0
    assert in_kstars(dout_star, k1) == 0


def test_out_kstars():
    k0 = 2
    k1 = 3

    assert out_kstars(dpath, k0) == 0
    assert out_kstars(dpath, k1) == 0
    assert out_kstars(din_star, k0) == 0
    assert out_kstars(din_star, k1) == 0
    assert out_kstars(dout_star, k0) == 10
    assert out_kstars(dout_star, k1) == 10


def test_mutuals():
    assert mutuals(mcustom) == 2
    assert mutuals(mdcustom) == 1


def test_stats_transform():
    decay = 0.5
    k0 = 2
    k1 = 3

    stats_comp = stats_transform([NEdges(), GWD(decay), KStars(k0)])
    np.testing.assert_almost_equal(
        stats_comp(path), np.array([3, 4.786938, 2]), 5
    )
    np.testing.assert_almost_equal(
        stats_comp(custom), np.array([7, 8.337899, 12]), 5
    )

    stats_comp = stats_transform([InKStars(k0), OutKStars(k1)])
    np.testing.assert_almost_equal(stats_comp(dpath), np.array([1, 0]))
    np.testing.assert_almost_equal(stats_comp(din_star), np.array([10, 0]))
