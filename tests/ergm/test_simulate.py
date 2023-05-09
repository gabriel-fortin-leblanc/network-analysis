import networkx as nx

from networkanalysis.ergm.simulate import *
from networkanalysis.statistics import *


def test_simulate():
    decay = 0.5
    k = 2
    ngraphs0 = 10
    ngraphs1 = 1000
    param0 = np.array([-1, 0.2, 0.5])
    param1 = np.array([-1, 1])
    stats_comp0 = stats_transform([NEdges(), GWD(decay), KStars(k)])
    stats_comp1 = stats_transform([NEdges(), Mutuals()])
    path = nx.path_graph(4)
    n = 50
    burnin = 1000
    thin = 3

    graphs = simulate(ngraphs0, param0, stats_comp0, n)
    assert len(graphs) == ngraphs0
    assert np.all([type(graph) is nx.Graph for graph in graphs])
    assert graphs[0] != graphs[-1]
    graphs = simulate(ngraphs0, param0, stats_comp0, path, burnin, thin)
    assert len(graphs) == ngraphs0
    assert np.all([type(graph) is nx.Graph for graph in graphs])
    assert graphs[0] != graphs[-1]
    graphs = simulate(ngraphs1, param1, stats_comp1, n)
    assert len(graphs) == ngraphs1
    assert np.all([type(graph) is nx.Graph for graph in graphs])
    assert graphs[0] != graphs[-1]
    graphs = simulate(ngraphs1, param1, stats_comp1, path, burnin, thin)
    assert len(graphs) == ngraphs1
    assert np.all([type(graph) is nx.Graph for graph in graphs])
    assert graphs[0] != graphs[-1]
    graphs, summary = simulate(
        ngraphs1, param1, stats_comp1, path, burnin, thin, True
    )
    assert type(summary["rate"]) is float
    assert 0 <= summary["rate"] and summary["rate"] <= 1
