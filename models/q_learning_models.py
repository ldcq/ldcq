import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class MLP_Q(nn.Module):
    '''
    Q(s,z) is our Abstract MLP Q function which takes as input current state s and skill z, 
    and outputs the expected return on executing the skill
    '''
    def __init__(self,state_dim,z_dim,h_dim=256):
        super(MLP_Q,self).__init__()

        z_embed_dim = h_dim//2
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU()
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(z_dim, z_embed_dim),
            nn.LayerNorm(z_embed_dim),
            nn.GELU(),
            nn.Linear(z_embed_dim, z_embed_dim),
            nn.LayerNorm(z_embed_dim),
            nn.GELU()
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(h_dim+z_embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self,s,z):
        '''
        INPUTS:
            s: batch_size x state_dim
            z: batch_size x z_dim
        OUTPUS:
            q_sz: batch_size x 1
        '''
        state_embed = self.state_mlp(s)
        z_embed = self.latent_mlp(z)
        s_z_cat = torch.cat([state_embed,z_embed], dim=1)
        q_sz = self.output_mlp(s_z_cat)
        return q_sz