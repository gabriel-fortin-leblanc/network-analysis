import networkx as nx
import pytest

from networkanalysis.ergm.likelihood import *
from networkanalysis.statistics import *

# Create undirected graphs.
path = nx.path_graph(4)
cycle = nx.cycle_graph(5)
complete = nx.complete_graph(4)
bi_complete = nx.complete_bipartite_graph(2, 3)
custom = nx.Graph()
custom.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5)])

# Create models.
decay = 0.5
k = 2
stats_comp0 = stats_transform([NEdges(), GWD(decay), KStars(k)])
stats_comp1 = stats_transform([NEdges(), Mutuals()])


class TestMPLE:
    @pytest.mark.parametrize(
        "graph, stats_comp",
        [
            (path, stats_comp0),
            (path, stats_comp1),
            (cycle, stats_comp0),
            (cycle, stats_comp1),
        ],
    )
    def test_mple(self, graph, stats_comp, benchmark):
        breakpoint()
        pass
        res = benchmark(mple(graph, stats_comp))
        assert type(res) is np.ndarray
