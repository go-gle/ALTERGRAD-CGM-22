import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class SkipMLP(nn.Module):
    def __init__(self, architecture=[16, 32, 64], n_nodes=50, cond_dim=7):
        super(SkipMLP, self).__init__()
        self.arch = architecture
        self.n_layers = len(self.arch)
        self.n_nodes = n_nodes
        self.mlp_layers = [nn.Linear(cond_dim, self.arch[0])]
        for i in range(self.n_layers - 2):
            self.mlp_layers.append(nn.Linear(self.arch[i], self.arch[i + 1]))

        #skip connection
        self.mlp_layers.append(nn.Linear(self.arch[-2] + self.arch[0], self.arch[-1]))
        self.mlp_layers.append(nn.Linear(self.arch[-1], 2*n_nodes*(self.n_nodes-1)//2))


        self.mlp = nn.ModuleList(self.mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.stats
        for i in range(len(self.mlp) - 1):
            #skip connection
            if i == len(self.mlp) - 2:
                x = torch.cat((x, skip), dim=1)

            x = self.relu(self.mlp[i](x))
            
            if i == 0:
                skip = x

        x = self.mlp[-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj
    
    def loss_function(self, data):
        adj = self.forward(data)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return recon

