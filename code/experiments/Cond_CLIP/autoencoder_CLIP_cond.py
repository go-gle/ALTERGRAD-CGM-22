import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

# Decoder (from Gleb's cond_autoencdoer.py)
class CondDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, cond_dim=32, cond_hid_dim=16):
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




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
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

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class CondCLIPVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, n_cond):
        super(CondCLIPVAE, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = CondDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.feature_encoder = FeatureEncoder(n_cond, latent_dim)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
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

    def contrastive_loss(self, graph_embeddings, feature_embeddings, temperature=0.07):
        """Compute CLIP-style contrastive loss between graph and feature embeddings"""
        # Normalize embeddings (CLIP thing)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        feature_embeddings = F.normalize(feature_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(graph_embeddings, feature_embeddings.T) / temperature

        # Labels are the diagonal elements (matching pairs)
        labels = torch.arange(similarity.shape[0], device=similarity.device)

        # Compute loss in both directions
        loss_g2f = F.cross_entropy(similarity, labels)
        loss_f2g = F.cross_entropy(similarity.T, labels)

        # Return mean of both directions
        return (loss_g2f + loss_f2g) / 2

    def compute_clip_loss(self, data, beta=0.0005):
        """
        Compute CLIP loss given a batch of data and feature encoder
        """
        # Get embeddings
        graph_embeddings = self.encode(data)
        feature_embeddings = self.feature_encoder(data)
        #compute loss
        #x_g  = self.encoder(data)
        #mu = self.fc_mu(x_g)
        #logvar = self.fc_logvar(x_g)
        #x_g = self.reparameterize(mu, logvar)

        clip_loss = self.contrastive_loss(graph_embeddings, feature_embeddings)
        #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = clip_loss # + beta*kld

        return loss# , clip_loss, kld
    
    def compute_clip_loss_from_x_g(self, x_g, data):
        feature_embeddings = self.feature_encoder(data)
        clip_loss = self.contrastive_loss(x_g, feature_embeddings)
        return clip_loss

    def recon_loss(self, data): 
        x_g = self.encode(data)
        cond = self.feature_encoder(data)
        adj = self.decoder(x_g, cond)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return recon
    
    def recon_loss_from_x_g(self, x_g, data):
        cond = self.feature_encoder(data)
        adj = self.decoder(x_g, cond)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return recon

    def joint_loss_function(self, data, scaling=1):
        x_g = self.encode(data)
        clip_loss = self.compute_clip_loss_from_x_g(x_g, data)
        recon_loss = self.recon_loss_from_x_g(x_g, data)
        loss = recon_loss + clip_loss * scaling # Clip loss is much bigger.

        return loss, recon_loss, clip_loss

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_feature_encoder(self):
        for param in self.feature_encoder.parameters():
            param.requires_grad = False

    def freeze_graph_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_logvar.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_feature_encoder(self):
        for param in self.feature_encoder.parameters():
            param.requires_grad = True

    def unfreeze_graph_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.fc_mu.parameters():
            param.requires_grad = True
        for param in self.fc_logvar.parameters():
            param.requires_grad = True
    
# Feature encoder
""" Brain dead MLP
Note that hidden dim has to be equal to the graph encoder's hidden dim
hence this number being big, but not as big as for the original feature encoder "cond_mlp"
from denoise_model.py that originally was 128 !

Please use the utils.py that standard scales the features for my mental health

This very encoder can be then reused to encode the features in the conditionning of the denoiser if we so 
choose to, as it theretically is a "better" encoder because it was trained to make sense"""
class FeatureEncoder(nn.Module):
    def __init__(self, n_cond, hidden_dim):
        super(FeatureEncoder, self).__init__()
        # 2 layers because why not
        self.mlp = nn.Sequential(
            nn.Linear(n_cond, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, data):
        if isinstance(data, torch.Tensor):
            x = data
            #print(x.shape)
        else:
            x = data.stats
        out = self.mlp(x)
        return out