from collections import namedtuple
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
import lightning.pytorch as pl

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DifficultyViT(pl.LightningModule):
    def __init__(self, *, vocab_size, resolutions, hidden_dim, depth, heads, mlp_dim, dim_head = 64, use_flash = True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.resolutions = resolutions

        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, use_flash)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patches):
        x = self.token_embedding(patches)

        b, _, dim = x.shape
        idx = 0
        
        # For storing the pos-encoded chunks to be concatenated later
        pos_encoded_chunks = []

        for resolution in self.resolutions:
            num_patches_for_res = resolution * resolution

            # Extract chunk from x for this resolution and reshape to (b, h, w, dim)
            chunk = x[:, idx : idx + num_patches_for_res, :]
            chunk = chunk.reshape(b, resolution, resolution, dim)

            # Compute sin-cos position embeddings in 2D
            pe = posemb_sincos_2d(chunk)  # should return shape (b, resolution * resolution, dim)

            # Flatten back to (b, h*w, dim)
            chunk = chunk.reshape(b, -1, dim)
            # Add position embeddings
            chunk = chunk + pe

            pos_encoded_chunks.append(chunk)
            idx += num_patches_for_res

        # Concatenate all chunks back along the sequence dimension
        x = torch.cat(pos_encoded_chunks, dim=1)  # (b, sum_of_all_r^2, dim)

        x = self.transformer(x)
        print(x.shape)

        return self.linear_head(x)
    
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = self.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer