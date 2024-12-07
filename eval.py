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
from extract_feats import extract_feats
import networkx as nx
import re
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from extract_feats import extract_numbers

#parsing
parser = argparse.ArgumentParser(description='Parses files to compare')

# Add arguments for two paths with default values
parser.add_argument('--path_to_truth', type=str, default='data/test/test.txt',
                    help='path/to/data/test/test.txt (default: data/test/test.txt')
parser.add_argument('--path_to_prediction', type=str, default='output.csv', 
                    help='path/to/output.csv (default: ./output.csv)')

# Parse the arguments
args = parser.parse_args()

truth_file = args.path_to_truth
pred_file = args.path_to_prediction

# get groundtruth features
y_test = []
with open(truth_file, 'r') as f:
    for line in f.readlines():
        splitted = line.split(',')
        index = splitted[0][6:]
        prompt = ','.join(splitted[1:])
        y_test.append([index] + extract_numbers(prompt))


y_test = pd.DataFrame(y_test, columns=['index', 'nodes', 'edges', 'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities'])
y_test.set_index('index', inplace=True)
y_test.sort_index(inplace=True)

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
    core_numbers = nx.core_number(G)
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
scaler = StandardScaler()
y_test_scaled = scaler.fit_transform(y_test)
y_pred_scaled = scaler.transform(y_pred)
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

print(f"Mean absolute error : {mae}")