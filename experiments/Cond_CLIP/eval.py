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
import torch
from extract_feats import extract_numbers
from utils import construct_nx_from_adj

# apply the pipeline and eval
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def eval(loader, autoencoder, latent_dim, cond_decoder=False):
    targets = []
    preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            stat = data.stats
            targets.append(stat.cpu().numpy())
            bs = stat.size(0)
            x_sample = torch.randn(bs, latent_dim, device=device)
            if not cond_decoder:
                adj = autoencoder.decode_mu(x_sample, stat)
            else:
                stat = autoencoder.feature_encoder(data)
                adj = autoencoder.decode_mu(x_sample, stat)
            for i in range(bs):
                Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
                preds.append(get_features(Gs_generated))

    preds = np.array(preds)
    targets = np.concatenate(targets)

    # get standard scaling stuff
    try:
        cond_mean, cond_std = torch.load('cond_standard_scaling')
    except Exception as e:
        print("\n\n--------------------\nPlease generate the standard scaling: \n\npython3 get_cond_scaler.py\n--------------------\n")
        raise e
    ###
    cond_mean, cond_std = cond_mean.numpy(), cond_std.numpy()
    #y_test_scaled = scaler.fit_transform(targets) I am a dumbass
    y_pred_scaled = (preds - cond_mean) / cond_std
    mae = mean_absolute_error(targets, y_pred_scaled)
    return mae

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
