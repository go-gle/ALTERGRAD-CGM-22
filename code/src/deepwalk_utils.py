"""this is an alternative to the default spectral embedding
This code is only useful to preprocess the graph data (train and val sets)

It builds upon what was done in  lab 5, deepwalk.py and node_classification.py"""


from random import randint
from gensim.models import Word2Vec

import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.baseline.extract_feats import extract_feats, extract_numbers

import torch.cuda

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    walk = [node]
    for i in range(walk_length): #ensures that a walk of length 1 is just the starting node to fit the indexation in the instruction sheet
        node = np.random.choice(list(G.neighbors(node))) # jump uniformly at random
        walk.append(node)

    walk = [str(node) for node in walk]
    return walk


# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    for i, node in enumerate(G.nodes()):
        walks_from_node = [random_walk(G, node, walk_length) for _ in range(num_walks)]
        walks += walks_from_node
        #if i % 1000 == 0:
            #print(f'{i} / {G.number_of_nodes()} done ...')

    permuted_walks = np.random.permutation(walks)
    return permuted_walks.tolist()

"""
# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.to(device)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
"""

# train the deepwalk for each graph
def train_global_deepwalk_embeddings(graph_path, n_dim, n_walks=10, walk_length=12):
    all_walks = []
    graph_node_mapping = {}
    
    files = [f for f in os.listdir(graph_path)]
    print(f'Generating {n_walks} walks of length {walk_length} per node for each graph ...')
    for fileread in tqdm(files):
        fread = os.path.join(graph_path, fileread)
        
        # Load graph
        if fileread.endswith(".graphml"):
            G = nx.read_graphml(fread)
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        else:
            G = nx.read_edgelist(fread)
        
        # Generate walks with global node mapping
        graph_walks = generate_walks(G, num_walks=n_walks, walk_length=walk_length)
        
        # Track node mapping for later embedding retrieval
        graph_node_mapping[fileread] = {str(old_id): str(new_id) for new_id, old_id in enumerate(G.nodes())}
        
        all_walks.extend(graph_walks)
    
    # Train global Word2Vec model
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=20, hs=1)
    model.build_vocab(all_walks)
    print('Training Word2Vec ...')
    model.train(all_walks, total_examples=model.corpus_count, epochs=5)
    
    return model, graph_node_mapping

def get_graph_embeddings(G, global_model, n_dim):
    embeddings = np.zeros((G.number_of_nodes(), n_dim))
    for i, node in enumerate(G.nodes()):
        #print(embeddings.shape)
        #print(global_model.wv[str(node)].shape)
        embeddings[i,:] = global_model.wv[str(node)]
    return embeddings



# now we modify the preprocess_dataset from the provided utils.py
def preprocess_dataset_deepwalk(dataset, n_max_nodes, emb_dim):

    data_lst = []
        
    filename = './data/dataset_'+dataset+'.pt'
    graph_path = './data/'+dataset+'/graph'
    desc_path = './data/'+dataset+'/description'
    if os.path.isfile(filename):
        data_lst = torch.load(filename)
        print(f'Dataset {filename} loaded from file')
    else:
        if os.path.isfile('deepwalk_model.pt'):
            print("Loading pre-trained Word2Vec model...")
            model = torch.load('deepwalk_model.pt')
        elif dataset == 'train':
            print("Generating and training Word2Vec model...")
            model, _ = train_global_deepwalk_embeddings(graph_path, n_dim=emb_dim)
            torch.save(model, 'deepwalk_model.pt')
            print('ok')

        # traverse through all the graphs of the folder
        files = [f for f in os.listdir(graph_path)]
        adjs = []
        n_nodes = []
        print(f'Generating {dataset} ...')
        for fileread in tqdm(files):
            tokens = fileread.split("/")
            idx = tokens[-1].find(".")
            filen = tokens[-1][:idx]
            extension = tokens[-1][idx+1:]
            fread = os.path.join(graph_path,fileread)
            fstats = os.path.join(desc_path,filen+".txt")
            #load dataset to networkx
            if extension=="graphml":
                G = nx.read_graphml(fread)
                # Convert node labels back to tuples since GraphML stores them as strings
                G = nx.convert_node_labels_to_integers(
                    G, ordering="sorted"
                )
            else:
                G = nx.read_edgelist(fread)
            # use canonical order (BFS) to create adjacency matrix
            ### BFS & DFS from largest-degree node
            
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            # rank connected componets from large to small size
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            node_list_bfs = []
            for ii in range(len(CGs)):
                node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                degree_sequence = sorted(
                node_degree_list, key=lambda tt: tt[1], reverse=True)
                bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_bfs += list(bfs_tree.nodes())
            adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
            adj = torch.from_numpy(adj_bfs).float()
            
            edge_index = torch.nonzero(adj).t()
            size_diff = n_max_nodes - G.number_of_nodes()
            
            x = torch.zeros(G.number_of_nodes(), emb_dim+1)
            x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)

            # Use pre-trained DeepWalk embeddings
            deepwalk_emb = get_graph_embeddings(G, model, emb_dim)

            mn = min(G.number_of_nodes(), emb_dim)
            mn += 1
            x[:,1:mn] = torch.from_numpy(deepwalk_emb[:,:emb_dim])

            adj = F.pad(adj, [0, size_diff, 0, size_diff])
            adj = adj.unsqueeze(0)
            feats_stats = extract_feats(fstats)
            feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
            data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            #print(data_lst)
            torch.save(data_lst, filename)
        print(f'Dataset {filename} saved')
    return data_lst


    