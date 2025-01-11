import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class CondDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, cond_dim=7, cond_hid_dim=5):
        super(CondDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.cond_layers = nn.Sequential(
            nn.Linear(cond_dim, cond_hid_dim), nn.ReLU(),
            nn.Linear(cond_hid_dim, cond_hid_dim), nn.ReLU()
            )

        self.latent_layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            )

        mlp_layers = [nn.Linear(hidden_dim + cond_hid_dim, hidden_dim)] + [nn.Linear(hidden_dim , hidden_dim) for i in range(n_layers-3)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        cond_emb = self.cond_layers(cond)
        latent_emb = self.latent_layer(x)
        emb = torch.concat((cond_emb, latent_emb), dim=1)
        for i in range(len(self.mlp) - 1):
            x = self.relu(self.mlp[i](emb))
        
        x = self.mlp[-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class CondGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, cond_dim=7, cond_hid_dim=5):
        super().__init__()
        self.dropout = dropout

        self.cond_layers = nn.Sequential(
            nn.Linear(cond_dim, cond_hid_dim), nn.BatchNorm1d(cond_hid_dim), nn.ReLU(),
            nn.Linear(cond_hid_dim, cond_hid_dim), nn.ReLU()
            )
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim + cond_hid_dim)
        self.fc = nn.Linear(hidden_dim + cond_hid_dim, latent_dim)
        

    def forward(self, data, cond):
        edge_index = data.edge_index
        x = data.x
        cond_emb =  self.cond_layers(cond)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = torch.concat((cond_emb, out), dim=1)
        out = self.bn(out)
        out = self.fc(out)
        return out


class CondVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, cond_dim=7, cond_hid_dim=5):
        super(CondVariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = CondGIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, cond_dim=cond_dim, cond_hid_dim=cond_hid_dim)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = CondDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, cond_dim=cond_dim, cond_hid_dim=cond_hid_dim)

    def forward(self, data, cond):
        x_g = self.encoder(data, cond)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data, cond):
        x_g = self.encoder(data, cond)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, cond):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g, cond)
       return adj

    def decode_mu(self, mu, cond):
       adj = self.decoder(mu, cond)
       return adj

    def loss_function(self, data, cond, beta=0.05):
        x_g  = self.encoder(data, cond)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
