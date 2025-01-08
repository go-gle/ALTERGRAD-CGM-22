import networkx as nx
import numpy as np


def get_features(G):
    """ returns a list of features from G.
    We consider by default the features ['nodes', 'edges',
    'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities']"""
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    A = nx.adjacency_matrix(G)
    degree = A.sum(axis=1).mean()
    triangles = len(nx.triangles(G))
    clust_coef = nx.average_clustering(G)
    core_numbers = nx.core_number(G)
    max_k_core = max(core_numbers.values()) if core_numbers else 0
    #the number of communities is computed with an algorithm. I am choosing the one that is imported in utils.py
    #note that the algorithm yields a number of community with randomness, hence the averaging over 10 attemps and the float
    communities = np.array([len(nx.community.louvain_communities(G)) for _ in range(10)]).mean()

    return [nodes, edges, degree, triangles, clust_coef, max_k_core, communities]