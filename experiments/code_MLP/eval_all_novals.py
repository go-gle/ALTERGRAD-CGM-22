import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from simple_MLP import MixMLP
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

from deepwalk_utils import preprocess_dataset_deepwalk
from utils_bert_encoding import preprocess_dataset_bert
from eval_final import eval


from torch.utils.data import Subset


"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")
#### CUSTOM ONE FOR CLIP

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-mlp', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--architecture', type=str, default='16,32,64', help="architecture, i.e. size of the mlp layers between the first and last ones")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-mlp', action='store_false', default=True, help="Flag to enable/disable MLP training (default: enabled)")


# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")


# CUSTOM ARGS
# pick data folder
parser.add_argument('--dataset', type=str, default="data", help="specifies path to dataset")

# toggle denosier usage
#early stopping rounds
parser.add_argument('--early-stopping-rounds', type=int, default=30, help="early stopping rounds patience (default: 15)")

#cutoff for the mixture
parser.add_argument('--cutoff', type=float, default=0., help="cutoff for the number of nodes (standardized) to split mixture of mlps")

#embedd graphs using deepwalk
parser.add_argument('--graph_embedding', type=str, default="spectral", help="specifies embedding. Possible values : 'Deepwalk', 'spectral'")

#embedd prompts using bert
parser.add_argument('--text_embedding', type=str, default="basic", help="specifies embedding. Possible values : 'bert', 'basic'")

# toggle conditionning in the denoiser
parser.add_argument('--resume-training-mlp', action='store_true', default=False, help="Flag to pick up the training of the autoencoder")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
if args.graph_embedding == 'deepwalk':
    trainset = preprocess_dataset_deepwalk("train", args.n_max_nodes, args.spectral_emb_dim)
    validset = preprocess_dataset_deepwalk("valid", args.n_max_nodes, args.spectral_emb_dim)
    testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)

elif args.text_embedding == 'bert':
    testset = preprocess_dataset_bert("test", args.n_max_nodes, args.spectral_emb_dim, args.n_condition)
    trainset = preprocess_dataset_bert("train", args.n_max_nodes, args.spectral_emb_dim, args.n_condition)
    validset = preprocess_dataset_bert("valid", args.n_max_nodes, args.spectral_emb_dim, args.n_condition)
    

else:
    #trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.dataset)
    #validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.dataset)
    testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.dataset)




test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# initialize MLP
architecture = [int(s) for s in args.architecture.split(',')]

# loop on what was produced
files = os.listdir('noval_runs/mlps/')

with open('noval_runs/progress.csv', 'a') as prog:
    prog.write('run,epoch,mae\n')
for file in tqdm(files):
    mlp = MixMLP(architecture, n_nodes=args.n_max_nodes, cutoff=args.cutoff).to(device)
    splitted = file.split('_')
    if len(splitted) == 2:
        continue
    run, epoch = splitted[1], splitted[2][2:-8]
    checkpoint = torch.load(f'noval_runs/mlps/{file}')
    mlp.load_state_dict(checkpoint['state_dict'])

    mlp.eval()

    # Save to a CSV file
    with open("noval_runs/output_csvs/temporary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])
        for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
            data = data.to(device)

            stat = data.stats
            graph_ids = data.filename
            adj = mlp(data)
            stat_d = torch.reshape(stat, (-1, args.n_condition))


            for i in range(stat.size(0)):
                stat_x = stat_d[i]

                Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
                stat_x = stat_x.detach().cpu().numpy()

                # Define a graph ID
                graph_id = graph_ids[i]

                # Convert the edge list to a single string
                edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
                # Write the graph ID and the full edge list as a single row
                writer.writerow([graph_id, edge_list_text])


    ### Evaluate
    truth_file = f'{args.dataset}/test/test.txt'
    pred_file = "noval_runs/output_csvs/temporary.csv"

    mae = eval(truth_file, pred_file)

    print(f"-----------------------------------------------\nTest MAE : {mae}\n-----------------------------------------------")


    os.rename("noval_runs/output_csvs/temporary.csv", f"noval_runs/output_csvs/MLP_{str(mae)[:8].replace('.', '-')}_noval{run}_ep{epoch}.csv")
    os.rename(f'noval_runs/mlps/{file}', f"noval_runs/mlps/MLP_{str(mae)[:8].replace('.', '-')}_noval{run}_ep{epoch}.pth.tar")
    with open('noval_runs/progress.csv', 'a') as prog:
        prog.write(f'{run},{epoch},{str(mae)[:10]}\n')