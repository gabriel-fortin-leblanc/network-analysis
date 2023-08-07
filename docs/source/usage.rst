Usage
=====

This documentation will suppose that you are aware of the basics of Python and the different packages used in this project, such as [NetworkX](https://networkx.org/) or [Numpy](https://numpy.org/). If you are not, you can find a lot of tutorials on the Internet, or even on the official websites of the packages.

The commands in this file are written for Linux systems. If you are using Windows, replace `python3` by `py`. Also, if you are using a virtual environment, you may need to replace `python3` by `python`.

The package Network Analysis is composed of different modules, each one containing different functions. The main modules are:

- `network.statistics` for the computation of different statistics on graphs;
- `network.ergm` which contains multiple submodules for analyzing exponential random graph models.

Installation
------------

If Python 3 is not installed on your computer, you can download it from the [official website](https://www.python.org/downloads/), or use your package manager if you are using Linux. The package Network Analysis can be installed from PyPI using the following command:

.. code-block:: bash
    
    python3 -m pip install network-analysis


To force to install a specific version of the package, you can use the following command:

.. code-block:: bash

    python3 -m pip install network-analysis==0.1.0


for the version 0.1.0 for example.

Statistics
----------

To use the package, you first need to import the specific modules you want to use. For example, if you want to compute the geometrically weighted degree of a graph, you can use the following command:

.. code-block:: python

    from network-analysis import statistics as stat

    # Create a graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

    # Compute the geometrically weighted degree
    stat.gwd(G)

You can also create a function that computes a list of sufficient statistics relative to your application. Let us say we want the sufficient statistics function for a two-star model, which is the number of edges, and the number of paths of length 2. We can create the following function:

.. code-block:: python

    ss = stat.StatsComp([ # For Statistics Computer
        stat.NEdges(),
        stat.KStars(2),
    ])

and then simply use it on a graph:

.. code-block:: python

    ss(G)

**Note:** For each statistics function, such as `gwd` or `mutuals`, that `network-analysis.statistics` contains, it exists a class that does the same thing, but that can be used in a `StatsComp` object. It allows you to specify once and for all some parameters, such as the number of stars for `KStars`, and then use it on any graph.

Exponential Random Graph Models
-------------------------------

The package Network Analysis also contains a lot of tools for exponential random graph models. For example, if you want to simulate a random graph from a two-star model, you can use the following command:

.. code-block:: python

    from network-analysis import ergm as ergm
    import numpy as np

    # Create a statistics computer
    ss = ergm.StatsComp([
        ergm.NEdges(),
        ergm.KStars(2),
    ])
    # Create a parameter
    theta = np.array([-0.5, 1])

    ergm.simulate(100, theta, ss)

The function `simulate` has multiple parameters to collect various information on Markov chain used to simulate the random graph. You can find more information on the documentation of the function.

Suppose now that you observed a graph `G` and you want to compute the maximum likelihood estimator of the parameter of a two-star model with the statistics compute `ss`. You can use the following command:

.. code-block:: python

    ergm.ml(G, ss)


The submodule `ergm.likelihood` contains multiple functions such as `ml` that can be used to compute the maximum likelihood estimator, `mpl` that computes the maximum pseudolikelihood, or `apl` for getting the adjusted pseudolikelihood function. For more information, you can refer to the documentation.
