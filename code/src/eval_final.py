""" Computes the MAE between the typical ouput (of the form of the default output.csv and)
 and the target values (extract the features with the provided extract_feats.py
 
 Please put this file in your code/ dir.

 By default, if the data/ dir is in the code/ dir like this .py and if
 the output.csv is also in the code/ dir, everything work. OTHERWISE, parse as so:
 
 python3 eval.py path/to/data/test/test.txt path/to/output.csv"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.baseline.extract_feats import extract_numbers

def eval(truth_file, pred_file):

    # get groundtruth features
    y_test = []
    with open(truth_file, 'r') as f:
        for line in f.readlines():
            splitted = line.split(',')
            index = int(splitted[0][6:])
            prompt = ','.join(splitted[1:])
            y_test.append([index] + extract_numbers(prompt))


    y_test = pd.DataFrame(y_test, columns=['index', 'nodes', 'edges', 'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities'])
    y_test.set_index('index', inplace=True)
    y_test.sort_index(inplace=True)

    # now let's load our predictions dataframe:
    y_pred = []

    with open(pred_file, 'r') as f:
        for line in f.readlines()[1:]:
            splitted = line.split(',')
            index = splitted[0][6:]
            graph_string = ','.join(splitted[1:])
            G = load_graph_from_string(graph_string)
            feats = get_features(G)
            y_pred.append([int(index)] + feats)

    y_pred = pd.DataFrame(y_pred, columns=['index', 'nodes', 'edges', 'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities'])
    y_pred.set_index('index', inplace=True)
    y_pred.sort_index(inplace=True)


    # they standard scaled the data (reversed engineered by yours truly)
     # get standard scaling stuff
    try:
        cond_mean, cond_std = torch.load('../../src/cond_standard_scaling')
    except Exception as e:
        print("\n\n--------------------\nPlease generate the standard scaling: \n\npython3 get_cond_scaler.py\n--------------------\n")
        raise e
    ###
    cond_mean, cond_std = cond_mean.numpy(), cond_std.numpy()
    y_test_scaled = (y_test - cond_mean) / cond_std 
    y_pred_scaled = (y_pred - cond_mean) / cond_std

    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

    return mae

# define a function that provided a graph G returns its features
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
    try:
        core_numbers = nx.core_number(G)
    except:
        print('passing 1...')
        core_numbers = 0
    max_k_core = max(core_numbers.values()) if core_numbers else 0
    #the number of communities is computed with an algorithm. I am choosing the one that is imported in utils.py
    #note that the algorithm yields a number of community with randomness, hence the averaging over 10 attemps and the float
    communities = np.array([len(nx.community.louvain_communities(G)) for _ in range(10)]).mean()

    return [nodes, edges, degree, triangles, clust_coef, max_k_core, communities]

#the graphs are going to be read from output.csv, which has a particular formatting.
def load_graph_from_string(graph_string):
    """ 
    Variable : line is a string like the ones that follow the first comma in output.csv
    Returns : a nx graph"""
    edges = re.findall(r'\((\d+), (\d+)\)', graph_string)
    G = nx.Graph()
    #this is a little cursed but here we are
    for edge in edges:
        G.add_edge(int(edge[0]), int(edge[1]))
    return G