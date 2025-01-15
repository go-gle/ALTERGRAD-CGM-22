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

import sys
original_sys_path_length = len(sys.path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from skip_MLP import SkipMLP
from src.denoise_model_custom import DenoiseNN, p_losses, sample
from src.baseline.utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

from src.deepwalk_utils import preprocess_dataset_deepwalk
from src.utils_bert_encoding import preprocess_dataset_bert
from src.eval_final import eval

sys.path = sys.path[:original_sys_path_length]

from torch.utils.data import Subset

#because we're running multiple scripts at once let's define an id at random not to mix up the files
id_ = np.random.randint(10000000)

#now fix the seed
#seed = 29
#np.random.seed(seed)
#torch.manual_seed(seed)

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
    trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.dataset)
    validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.dataset)
    testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.dataset)




# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# initialize MLP
architecture = [int(s) for s in args.architecture.split(',')]

mlp = SkipMLP(architecture, n_nodes=args.n_max_nodes).to(device)

if args.resume_training_mlp:
    print('Loading MLP checkpoint...')
    checkpoint = torch.load('MLP.pth.tar')
    mlp.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
early_stopping_counts = 0



#### TRAIN MLP
print("\nTraining the MLP....")
if args.train_mlp:

    best_val_loss = np.inf
    
    for epoch in range(1, args.epochs_mlp + 1):
        mlp.train()
        train_loss_all = 0
        train_count = 0
        cnt_train=0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = mlp.loss_function(data)
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()
        mlp.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0
        
        for data in val_loader:
            data = data.to(device)
            loss = mlp.loss_function(data)
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1
        
        if epoch % 1 == 0:
            print('Epoch: {:04d}, Train Reconstruction Loss: {:.5f}, Val Reconstruction Loss: {:.5f}'.format(epoch, train_loss_all/cnt_train, val_loss_all/cnt_val))
        
        scheduler.step()
        early_stopping_counts += 1
        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': mlp.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, f'temp/MLP_{id_}.pth.tar')
            early_stopping_counts = 0

        if early_stopping_counts == args.early_stopping_rounds:
            print('early stopping')
            break

    checkpoint = torch.load(f'temp/MLP_{id_}.pth.tar')
    mlp.load_state_dict(checkpoint['state_dict'])
else:
    checkpoint = torch.load('MLP.pth.tar')
    mlp.load_state_dict(checkpoint['state_dict'])


mlp.eval()

del train_loader, val_loader




# Save to a CSV file
with open(f"temp/MLP_{id_}.csv", "w", newline="") as csvfile:
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
pred_file = f"temp/MLP_{id_}.csv"

mae = eval(truth_file, pred_file)

print(f"-----------------------------------------------\nTest MAE : {mae}\n-----------------------------------------------")


os.rename(f"temp/MLP_{id_}.csv", f"runs/output_csvs/MLP_{str(mae)[:7].replace('.', '-')}.csv")

# save vae
if args.train_mlp:
    checkpoint = torch.load(f'temp/MLP_{id_}.pth.tar')
    mlp.load_state_dict(checkpoint['state_dict'])
    torch.save({
                    'state_dict': mlp.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, f"runs/mlps/MLP_{str(mae)[:7].replace('.', '-')}.pth.tar")

else:
    checkpoint = torch.load('MLP.pth.tar')
    mlp.load_state_dict(checkpoint['state_dict'])
    torch.save({
                    'state_dict': mlp.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, f"runs/mlps/MLP_{str(mae)[:7].replace('.', '-')}.pth.tar")

#save_args not to get lost
def save_args_to_file(args, file_path=f"runs/args/args_{str(mae)[:7].replace('.', '-')}.txt"):
    with open(file_path, "w") as f:
        f.write("Parsed Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments saved to {os.path.abspath(file_path)}")

save_args_to_file(args)