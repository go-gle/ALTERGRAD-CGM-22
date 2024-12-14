import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extract_feats import extract_feats
import networkx as nx
import re
from tqdm import tqdm
from random import choice

import argparse

parser = argparse.ArgumentParser(description='Parses arguments for generation')

# Add arguments for two paths with default values
parser.add_argument('--path-to-data', type=str, default='data/',
                    help='path to the data/ folder that itself has at least a train/ and val/ dir')
parser.add_argument('--n-switches', type=int, default=2, 
                    help='Number of graphs to generate by toggling edges')
parser.add_argument('--n-additions', type=int, default=2, 
                    help='Number of graphs to generate by adding nodes and edges')
parser.add_argument('--n-deletions', type=int, default=2, 
                    help='Number of graphs to generate by deleting nodes')
# Parse the arguments
args = parser.parse_args()

train_path = os.path.join(args.path_to_data, 'train/')
valid_path = os.path.join(args.path_to_data, 'valid/')
n_switch, n_add, n_del = args.n_switches, args.n_additions, args.n_deletions
# utils

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
    communities = int(np.round(communities))
    return [nodes, edges, degree, triangles, clust_coef, max_k_core, communities]

# Function to write a prompt
def write_prompt(features):
    """replaces the numbers by a string with the corresponding feature"""
    prompt1 = 'In this graph, there are XXX nodes connected by XXX edges. On average, each node is connected to XXX other nodes. Within the graph, there are XXX triangles, forming closed loops of nodes. The global clustering coefficient is XXX. Additionally, the graph has a maximum k-core of XXX and a number of communities equal to XXX.'
    prompt2 = "This graph comprises XXX nodes and XXX edges. The average degree is equal to XXX and there are XXX triangles in the graph. The global clustering coefficient and the graph's maximum k-core are XXX and XXX respectively. The graph consists of XXX communities."
    prompt = choice((prompt1, prompt2))
    
    chunks = prompt.split('XXX')
    result = ''
    for i, feature in enumerate(features):
        result += chunks[i] + str(feature)
    
    return result + chunks[-1]

def switch(G, n):
    """ Takes a graph G, and switches n (int) connections (0 to 1 or 1 to 0)
    from the adj matrix at random.
    Returns a new graph sligthly modified"""

    n_nodes = G.number_of_nodes()
    adj = nx.adjacency_matrix(G).toarray()
    for _ in range(n):
        i = j = 0 
        while i == j: # making sure i != j
            i, j = np.random.randint(0, n_nodes, 2)

        adj[i, j] = np.abs(adj[i, j] - 1)
        adj[j, i] = np.abs(adj[j, i] - 1)

    return nx.from_numpy_array(adj)

def add(G, n):
    """ takes a graph G and adds n nodes and connects them at random with existing nodes
    To determine the new connections, we take an existing node in G at random and look at its number of connections
    """
    n_nodes = G.number_of_nodes()
    adj = nx.adjacency_matrix(G).toarray()

    for iteration in range(n):
        #pick a node at random and select its degree
        random_node = np.random.randint(0, n_nodes)
        n_connections = adj[random_node].sum()
        # create the new line and column to insert in the adjacency mat
        new = np.random.randint(0, 2, (1, n_nodes + iteration))
        #add line
        adj = np.concatenate((adj,new), axis=0)
        # add col
        new = np.concatenate((new, np.zeros((1, 1))), axis=1)
        adj = np.concatenate((adj,new.T), axis=1)
    
    return nx.from_numpy_array(adj)

def delete(G, n):
    """takes G and returns a new graph with n nodes reomved.
    They're selected at random"""
    n_nodes = G.number_of_nodes()
    adj = nx.adjacency_matrix(G).toarray()

    for i in range(n):
        node = np.random.randint(0, n_nodes - i)
        adj = np.concatenate((adj[:node], adj[node+1:]), axis=0)
        adj = np.concatenate((adj[:, :node], adj[:, node+1:]), axis=1)
    return nx.from_numpy_array(adj)


import multiprocessing

def process_graph_file(arguments):
    """
    Process a single graph file with all transformations
    
    Args:
    - args: tuple containing (graphs_dir, desc_dir, filename)
    """
    graphs_dir, desc_dir, filename = arguments
    
    fread = os.path.join(graphs_dir, filename)
    
    # Load graph
    if filename.endswith(".graphml"):
        G = nx.read_graphml(fread)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    else:
        G = nx.read_edgelist(fread)
    
    # Extract index of the graph
    index = filename.split('.')[0][6:]
    
    # Number of nodes
    n_nodes = G.number_of_nodes()
    
    # Generate transformed graphs
    switches = [switch(G, np.random.randint(1, int(n_nodes * 0.2 + 1))) for _ in range(n_switch)]
    adds = [add(G, np.random.randint(1, int(n_nodes * 0.2 + 1))) for _ in range(n_add)]
    dels = [delete(G, np.random.randint(1, int(n_nodes * 0.2 + 1))) for _ in range(n_del)]
    new_Gs = switches + adds + dels
    
    # Save valid transformed graphs
    results = []
    for i, H in enumerate(new_Gs):
        if H.number_of_nodes() > 50 or H.number_of_nodes() < 10:
            continue
        elif H.number_of_edges() < 1:
            continue
        new_name = f'graph_{index}0000{i}'
        new_path_graph = os.path.join(graphs_dir, new_name + '.edgelist')
        new_path_desc = os.path.join(desc_dir, new_name + '.txt')
        
        nx.write_edgelist(H, new_path_graph)
        
        feats = get_features(H)
        prompt = write_prompt(feats)
        
        with open(new_path_desc, 'w') as f:
            f.write(prompt)
        
        results.append(new_name)
    
        #return results

def gen_from_listdir(set_dir, num_cores=None):
    """
    Generate graph transformations using multiprocessing
    
    Args:
    - set_dir: Directory containing graph files
    - num_cores: Number of cores to use (defaults to all available)
    """
    graphs_dir = os.path.join(set_dir, 'graph/')
    desc_dir = os.path.join(set_dir, 'description/')
    
    # Get list of files
    ls = os.listdir(graphs_dir)
    
    # Prepare arguments for each file
    file_args = [(graphs_dir, desc_dir, filename) for filename in ls]
    
    # Use all available cores if not specified
    if num_cores is None:
        num_cores = os.cpu_count()
    
    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress 
        list(tqdm(pool.imap(process_graph_file, file_args), total=len(file_args)))

# Usage
if __name__ == '__main__':
    print('Generating training examples ....')
    gen_from_listdir('data/train/', num_cores=None)  # Specify number of cores
    
    print('Generating validation samples ....')
    gen_from_listdir('data/valid/', num_cores=None)