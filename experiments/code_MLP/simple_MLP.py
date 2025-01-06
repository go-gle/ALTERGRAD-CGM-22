import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class SimpleMLP(nn.Module):
    def __init__(self, architecture=[16, 32, 64], n_nodes=50, cond_dim=7, mixing=False):
        super(SimpleMLP, self).__init__()
        self.arch = architecture
        self.n_layers = len(self.arch)
        self.n_nodes = n_nodes
        self.mixing = mixing

        self.mlp_layers = [nn.Linear(cond_dim, self.arch[0])]
        for i in range(self.n_layers - 1):
            self.mlp_layers.append(nn.Linear(self.arch[i], self.arch[i + 1]))

        self.mlp_layers.append(nn.Linear(self.arch[-1], 2*n_nodes*(self.n_nodes-1)//2))

        self.mlp = nn.ModuleList(self.mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        if self.mixing:
            x = data
        else:
            x = data.stats
        #print(x[:, 0])
        for i in range(len(self.mlp) - 1):
            x = self.relu(self.mlp[i](x))
        
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

class MixMLP(nn.Module):
    def __init__(self, architecture=[16, 32, 64], n_nodes=50, cond_dim=7, cutoff=0):
        super(MixMLP, self).__init__()
        self.arch = architecture
        self.n_nodes = n_nodes
        self.small_mlp = SimpleMLP(self.arch, n_nodes=n_nodes, mixing=True)
        self.big_mlp = SimpleMLP(self.arch, n_nodes=n_nodes, mixing=True)
        self.cutoff = cutoff

    def forward(self, data):
        x = data.stats
        mask_big = x[:, 0] > self.cutoff  
        mask_small = ~mask_big  

        # Split
        x_big = x[mask_big]
        x_small = x[mask_small] 

        output_big = self.big_mlp(x_big) if x_big.size(0) > 0 else torch.Tensor([]) 
        output_small = self.small_mlp(x_small) if x_small.size(0) > 0 else torch.Tensor([])

        # Reconstruct
        output = torch.zeros(x.shape[0], self.n_nodes, self.n_nodes).to(x.device)
        output[mask_big] = output_big 
        output[mask_small] = output_small
        return output

    def loss_function(self, data):
        adj = self.forward(data)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return recon