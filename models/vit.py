from collections import namedtuple
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
import lightning.pytorch as pl
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from models.vqgan import VQMultiModel
from typing import Union
from torchvision.transforms import v2
import torchvision
from torch.optim import lr_scheduler
# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# blocks = torch.cat([
#     torch.zeros(1),  # 1x1 resolution
#     torch.ones(4),  # 2x2 resolution 
#     torch.full((16,), 2),  # 4x4 resolution
#     torch.full((64,), 3),  # 8x8 resolution
#     torch.full((512,), 4)  # 16x16 resolution
# ], dim=0).to("cuda")
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
        
        # self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=341, KV_LEN=341)
        
        # print(self.block_mask)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        # out = flex_attention(q, k, v, block_mask=self.block_mask)

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
    
class AdaLNTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AdaLNAttention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                AdaLNFeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, cond):
        for attn, ff in self.layers:
            x = attn(x, cond) + x
            x = ff(x, cond) + x
        return x 
    

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob={self.drop_prob})'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, dim):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.dim = dim
        self.ln_wo_grad = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 2*dim))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.dim).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
    
class AdaLNAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True, drop_p=0, causal=None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        lin = nn.Linear(dim, 3*dim)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        # Pre-compute causal mask once
        block_mask = torch.zeros((341, 341))
        blocks = torch.cat([
            torch.zeros(1),  # 1x1 resolution
            torch.ones(4),  # 2x2 resolution 
            torch.full((16,), 2),  # 4x4 resolution
            torch.full((64,), 3),  # 8x8 resolution
            torch.full((256,), 4)  # 16x16 resolution
        ], dim=0)
        block_mask = blocks.unsqueeze(0) <= blocks.unsqueeze(1)
                
        self.register_buffer("block_mask", block_mask)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
    def causal(self, b, h, q_idx, kv_idx):
        
        blocks = torch.cat([
            torch.zeros(1),  # 1x1 resolution
            torch.ones(4),  # 2x2 resolution 
            torch.full((16,), 2),  # 4x4 resolution
            torch.full((64,), 3),  # 8x8 resolution
            torch.full((512,), 4)  # 16x16 resolution
        ], dim=0).to("cuda")
        # Hardcoded boundaries based on resolutions [1,2,4,8,16]
        # boundaries[i+1] = boundaries[i] + res[i]^2
        return blocks[q_idx] >= blocks[kv_idx]

    def forward(self, x, cond):
        gamma1, scale1, shift1 = self.ada_lin(cond).view(-1, 1, 3, self.dim).unbind(2)
        x = self.norm(x).mul(scale1.add(1)).add_(shift1)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.block_mask)
        # out = flex_attention(q, k, v, block_mask=self.block_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) * gamma1

class AdaLNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop_p=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 3*dim))
    def forward(self, x, cond):
        gamma2, scale2, shift2 = self.ada_lin(cond).view(-1, 1, 3, self.dim).unbind(2)
        x = self.norm(x).mul(scale2.add(1)).add_(shift2)
        return self.drop_path(self.net(x).mul(gamma2))

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
        
        output = self.linear_head(x).squeeze(dim=-1)

        return output
    
    
    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        difficulties = batch["difficulties"]
        # training_step defines the train loop.
        x = self.forward(tokens)
        
        # Create a mask for non-zero difficulties and tokens
        mask = (difficulties != 0) & (tokens != 0)
        
        # Compute the MSE loss only for the masked elements
        loss = F.mse_loss(x[mask], difficulties[mask])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        difficulties = batch["difficulties"]
        x = self.forward(tokens)
        mask = (difficulties != 0) & (tokens != 0)
        loss = F.mse_loss(x[mask], difficulties[mask])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
class NDPViT(pl.LightningModule):
    def __init__(self, *, vocab_size, num_classes=1000, resolutions, hidden_dim, depth, heads, mlp_dim, dim_head = 64, use_flash = True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.resolutions = resolutions

        self.transformer = AdaLNTransformer(hidden_dim, depth, heads, dim_head, mlp_dim, use_flash)
        self.class_token_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        self.level_embeddings = nn.Embedding(5, hidden_dim)

        self.ada_ln_before_head = AdaLNBeforeHead(hidden_dim)

        self.linear_head = nn.Linear(hidden_dim, vocab_size)
        
        self.learning_rate = 1e-4
        
        self.LEVEL_RANGES = [
            (0, 1),    # Level 1: class token
            (1, 5),    # Level 2: 2x2 tokens
            (5, 21),   # Level 3: 4x4 tokens
            (21, 85),  # Level 4: 8x8 tokens
            (85, 341)  # Level 5: 16x16 tokens
        ]

        # self.difficulty = DifficultyViT.load_from_checkpoint("./imagenet_256/diffvit/last.ckpt", vocab_size=9216, resolutions=[1, 2, 4, 8, 16], hidden_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=128)
        # self.difficulty.to("cuda")
        # self.difficulty.eval()
        
        self.tokenizer = VQMultiModel.load_from_checkpoint("./imagenet_256/vqgan_32768_16/epoch=39-step=480464.ckpt", n_embed=8192)
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.tokenizer.to("cuda")
        self.tokenizer.eval()
        self.tokenizer = torch.compile(self.tokenizer)
        
        blocks = torch.cat([
            torch.zeros(1, dtype=torch.int64),  # 1x1 resolution
            torch.ones(4, dtype=torch.int64),  # 2x2 resolution 
            torch.full((16,), 2, dtype=torch.int64),  # 4x4 resolution
            torch.full((64,), 3, dtype=torch.int64),  # 8x8 resolution
            torch.full((256,), 4, dtype=torch.int64)  # 16x16 resolution
        ], dim=0)
        
        # Create adjacency on cuda first
        T = 341
        adjacency = torch.full((T, 4), -1, dtype=torch.long)
        for i in range(T):
            inds = self.get_subpatch_indices(i)
            for j, sp_i in enumerate(inds):
                adjacency[i, j] = sp_i
        
        # Register as buffer and move to current device
        self.register_buffer('adjacency', adjacency, persistent=False)
        self.register_buffer('blocks', blocks, persistent=False)


    def find_level(self, i: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Returns which hierarchical level (0-5) the token index belongs to.
        
        Args:
            i: Either an integer index or tensor of indices
            
        Returns:
            Either an integer level or tensor of levels
        """
        if isinstance(i, torch.Tensor):
            levels = torch.zeros_like(i)
            for level, (start, end) in enumerate(self.LEVEL_RANGES):
                mask = (i >= start) & (i < end)
                levels[mask] = level
            return levels
        else:
            for level, (start, end) in enumerate(self.LEVEL_RANGES):
                if start <= i < end:
                    return level
            return -1
        
    def get_subpatch_indices(self, i: int) -> list:
        """Returns indices of 4 sub-patches in next level for given token index."""
        level = self.find_level(i)
        if level >= 4:
            return []
            
        # Get position in current level's grid
        start = self.LEVEL_RANGES[level][0]
        offset = i - start
        curr_res = 2 ** level  # Current resolution (1, 2, 4, 8, 16)
        row = offset // curr_res
        col = offset % curr_res
        
        # Get corresponding positions in next level's grid
        next_res = curr_res * 2
        next_start = self.LEVEL_RANGES[level + 1][0]
        row_start = row * 2
        col_start = col * 2
        
        # Return the 4 corresponding indices in next level
        indices = []
        for r in range(2):
            for c in range(2):
                next_row = row_start + r
                next_col = col_start + c
                next_idx = next_start + (next_row * next_res + next_col)
                indices.append(next_idx)
                
        return indices

    def forward(self, tokens):
        
                
        class_token = tokens[:, 0]
        tokens = tokens[:, 1:]
        
        patch_embeddings = self.token_embedding(tokens)
        class_embeddings = self.class_token_embedding(class_token)
        
        x = torch.cat([class_embeddings.unsqueeze(1), patch_embeddings], dim=1)
        
        # Create level embeddings based on which hierarchical level each token belongs to
        # Repeat for batch dimension and add class token level
        levels = self.blocks.repeat(x.shape[0], 1)
        
        # Add level embeddings to token embeddings
        level_embeddings = self.level_embeddings(levels)
        x = x + level_embeddings

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
            if resolution > 1:
                chunk = chunk + pe

            pos_encoded_chunks.append(chunk)
            idx += num_patches_for_res

        # Concatenate all chunks back along the sequence dimension
        x = torch.cat(pos_encoded_chunks, dim=1)  # (b, sum_of_all_r^2, dim)

        x = self.transformer(x, class_embeddings)
        x = self.ada_ln_before_head(x, class_embeddings)
        output = self.linear_head(x)

        return output
    
    def tokenize(self, images):
        
      
        with torch.no_grad():
            images = images.to(self.tokenizer.device, dtype=self.tokenizer.dtype)
            quant1, quant2, quant3, quant4, diff, info = self.tokenizer.encode(images)
            # info is typically: ((_, _, info1), (_, _, info2), (_, _, info3), (_, _, info4))
            (_, _, info1), (_, _, info2), (_, _, info3), (_, _, info4) = info
        
            tokens = torch.cat([
                (info4.reshape(images.shape[0], -1) + 64),
                (info3.reshape(images.shape[0], -1) + 64+8192),
                (info2.reshape(images.shape[0], -1) + 64+8192+8192),
                (info1.reshape(images.shape[0], -1) + 64+8192+8192+8192),
            ], dim=1)
            
        
        # if torch.randint(0, 2500, (1,)).item() == 0:
        #     randint = torch.randint(0, 1000000, (1,)).item()
        #     torchvision.utils.save_image(
        #         images, 
        #         f'/mnt/ndp/debug/encoded_sample_{randint}.png',
        #         nrow=int(images.shape[0] ** 0.5)  # Make a square grid
        #     )
        #     xrec = self.tokenizer.decode(quant1, quant2, quant3, quant4)
        #     torchvision.utils.save_image(
        #         xrec,
        #         f'/mnt/ndp/debug/decoded_sample_{randint}.png',
        #         nrow=int(xrec.shape[0] ** 0.5)  # Make a square grid
        #     )
        
        return tokens
        
    def ndp_step(self, tokens, detail_idx=None):
        """
        Next-Detail Prediction (NDP) step using masked-language-model-style loss.

        Args:
            tokens: (B, T)
            labels: (B,)
            detail_idx: Optional tensor of indices to use. If None, random indices are generated.

        Returns:
            loss: The computed cross entropy loss
        """
        B, T = tokens.shape

        # We'll keep a clone of tokens so original isn't overwritten
        tokens_modified = tokens.clone()

        # This mask will be True in positions we want to compute the cross-entropy over
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        # Calculate weights based on number of tokens in each level
        level_sizes = [end - start for start, end in self.LEVEL_RANGES[:-1]]  # Exclude level 4
        weights = torch.tensor(level_sizes, device=tokens.device, dtype=torch.float)
        weights = weights / weights.sum()  # Normalize to probabilities
        
        # Sample levels according to token count weights
        curr_level = torch.multinomial(weights.expand(B, -1), 1).squeeze(-1)
        
        # Handle per-batch masking based on each sample's curr_level
        for i in range(B):
            c_level = int(curr_level[i].item())
            
            # Create mask for all patches at higher levels
            for lvl in range(c_level + 2, 5):  # Assuming 5 levels total (0-4)
                lvl_start, lvl_end = self.LEVEL_RANGES[lvl]
                tokens_modified[i, lvl_start:lvl_end] = 0

            # For the next level, randomly mask some patches (~10%)
            curr_start, curr_end = self.LEVEL_RANGES[c_level + 1]
            curr_patches = tokens_modified[i, curr_start:curr_end]
            curr_mask = torch.rand_like(curr_patches.float()) < 0.3
            curr_patches[curr_mask] = 0

        # For each batch element, get indices of current level, select adjacencies, and apply
        for i in range(B):
            c_level = int(curr_level[i].item())
            curr_start, curr_end = self.LEVEL_RANGES[c_level]
            curr_indices = torch.arange(curr_start, curr_end, device=tokens.device)

            # Randomly select 80% of the current level indices
            num_to_select = max(1, int(0.8 * len(curr_indices)))
            selected = curr_indices[torch.randperm(len(curr_indices))[:num_to_select]]

            # Get adjacencies (shape might differ for each batch element)
            batch_adjacencies = self.adjacency[selected]

            # Apply to tokens_modified where adjacency is valid
            valid_mask = batch_adjacencies != -1
            tokens_modified[i, batch_adjacencies[valid_mask]] = 1
            loss_mask[i, batch_adjacencies[valid_mask]] = True
            
        
        
        # 4) Forward pass on the modified tokens
        logits = self.forward(tokens_modified)  # shape: (B, T, vocab_size)
        # Create targets with -100 for non-masked positions (ignored by cross_entropy)
        masked_targets = torch.full_like(tokens, -100)
        masked_targets[loss_mask] = tokens[loss_mask]      # Fill in actual targets for masked positions
        
        if self.trainer.global_step % 10000 == 0:
            print(f"Current step: {self.trainer.global_step}")
            print(f"Tokens modified: {tokens_modified[0]}")
            print("Masked targets", masked_targets[0])

        return F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1))
        
    def training_step(self, batch, batch_idx):
        
        labels = batch["label"]
        labels = torch.where(labels == -1,
                           torch.tensor(1000, device=labels.device),
                           labels)
        tokens = self.tokenize(batch["image"])
        
        # print("Tokens", tokens)
        
        # Randomly set 10% of the batch to [2]
        # Only modify first token in sequence
        # Generate random values on same device as tokens
        rand_vals = torch.rand(tokens.shape[0], device=tokens.device)
        labels = torch.where(rand_vals < 0.1,
                                          torch.tensor(1000, device=labels.device),
                                          labels)
        tokens = torch.cat([labels.unsqueeze(1), tokens], dim=1)
        loss = self.ndp_step(tokens)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Use deterministic indices based on batch size
        labels = batch["label"]
        labels = torch.where(labels == -1,
                           torch.tensor(1000, device=labels.device),
                           labels)
        tokens = self.tokenize(batch["image"])
        
        B = tokens.shape[0]
        detail_idx = torch.arange(0, B, device=tokens.device) % 85
        tokens = torch.cat([labels.unsqueeze(1), tokens], dim=1)
        loss = self.ndp_step(tokens, detail_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        trainable_params = []
        trainable_params.extend(self.transformer.parameters())
        trainable_params.extend(self.token_embedding.parameters())
        trainable_params.extend(self.class_token_embedding.parameters())
        trainable_params.extend(self.level_embeddings.parameters())
        trainable_params.extend(self.ada_ln_before_head.parameters())
        trainable_params.extend(self.linear_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=181)
        return [optimizer], [scheduler]
    
    def on_train_epoch_start(self, *args, **kwargs):
        if self.trainer.current_epoch == 90:
            print("Changing learning rate")
            optimizer = self.optimizers()
            optimizer.param_groups[0]['lr'] = self.learning_rate
    
    def log_images(self, batch, **kwargs):
        
        tokens = torch.zeros((1, 340), dtype=torch.long, device=self.device)
        uncond_tokens = torch.zeros((1, 340), dtype=torch.long, device=self.device)
        labels = torch.randint(1, 11, (1,), device=self.device)
        uncond_labels = torch.full_like(labels, 1000)
        tokens = torch.cat([labels.unsqueeze(1), tokens], dim=1)

        uncond_tokens = torch.cat([uncond_labels.unsqueeze(1), uncond_tokens], dim=1)
        idx = 0

        def generate_next_detail(tokens, level_idx):
            # Get all indices for current level
            level_start, level_end = self.LEVEL_RANGES[level_idx]
            available_indices = []
            for i in range(level_start, level_end):
                subpatch_indices = self.get_subpatch_indices(i)
                if torch.all(tokens[0, subpatch_indices] == 0) and tokens[0, i] != 0:
                    available_indices.append(i)
                    
            if not available_indices:
                return tokens, level_idx + 1  # Move to next level
                
            # Randomly select an index from available positions
            idx = available_indices[torch.randint(0, len(available_indices), (1,)).item()]
            
            # Get subpatch indices for selected idx
            subpatch_indices = self.get_subpatch_indices(idx)
            
            # Mark subpatch indices as masked
            tokens[0, subpatch_indices] = 1
            
            # Forward pass for conditioned and unconditioned tokens
            logits = self.forward(tokens.to("cuda"))
            uncond_logits = self.forward(uncond_tokens.to("cuda"))

            # Merge conditioned and unconditioned logits
            guidance_scale = 2.0
            logits = uncond_logits + guidance_scale * (logits - uncond_logits)

            # Determine valid token range based on hierarchical level
            if level_idx == 0:  # 2x2 level
                valid_range = (64, 64 + 8192)
            elif level_idx == 1:  # 4x4 level
                valid_range = (64 + 8192, 64 + 8192 + 8192)
            elif level_idx == 2:  # 8x8 level
                valid_range = (64 + 8192 + 8192, 64 + 8192 + 8192 + 8192)
            else:  # 16x16 level
                valid_range = (64 + 8192 + 8192 + 8192, 64 + 8192 + 8192 + 8192 + 8192)

            # Mask logits outside the valid range
            logits_masked = logits[0, subpatch_indices].clone()
            logits_masked[:, :valid_range[0]] = float('-inf')
            logits_masked[:, valid_range[1]:] = float('-inf')

            # Temperature sampling
            temperature = 0.5
            logits_masked = logits_masked / temperature

            # Top-k sampling
            k = 200
            probs = torch.softmax(logits_masked, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
            sampled_tokens = top_k_indices[torch.arange(len(sampled_indices)), sampled_indices]

            # Place the sampled tokens
            tokens[0, subpatch_indices] = sampled_tokens.to(dtype=tokens.dtype, device=tokens.device)

            return tokens, level_idx

        output = []
        level_idx = 0  # Start from level 0 (class token)
        total_details = 0
        
        while level_idx < 5 and total_details < 64:  # 5 is number of levels
            tokens, new_level_idx = generate_next_detail(tokens, level_idx)
            
            if new_level_idx != level_idx:  # We've completed current level
                level_idx = new_level_idx
            
            total_details += 1
            
            if total_details in [1, 4, 8, 16, 32, 64]:  # Plot at 1, 4, 8, 16, 32, 64 details
                with torch.no_grad():
                    # Handle final offsets for each level
                    level1_tokens = torch.where(tokens[0, 1:5] != 0, tokens[0, 1:5] - 64, tokens[0, 1:5])
                    level2_tokens = torch.where(tokens[0, 5:21] != 0, tokens[0, 5:21] - (64 + 8192), tokens[0, 5:21])
                    level3_tokens = torch.where(tokens[0, 21:85] != 0, tokens[0, 21:85] - (64 + 8192 + 8192), tokens[0, 21:85])
                    level4_tokens = torch.where(tokens[0, 85:341] != 0, tokens[0, 85:341] - (64 + 8192 + 8192 + 8192), tokens[0, 85:341])

                    # Convert to codebook entries
                    q1 = self.tokenizer.quantize_1.get_codebook_entry(level4_tokens.to("cuda"), (1, 16, 16, 8))
                    q2 = self.tokenizer.quantize_2.get_codebook_entry(level3_tokens.to("cuda"), (1, 8, 8, 8))
                    q3 = self.tokenizer.quantize_3.get_codebook_entry(level2_tokens.to("cuda"), (1, 4, 4, 8))
                    q4 = self.tokenizer.quantize_4.get_codebook_entry(level1_tokens.to("cuda"), (1, 2, 2, 8))

                    # Decode image
                    img = self.tokenizer.decode(q1, q2, q3, q4)
                    output.append(img)

        return output
                    



# class NDPViT(pl.LightningModule):
#     def __init__(self, *, vocab_size, resolutions, hidden_dim, depth, heads, mlp_dim, dim_head = 64, use_flash = True):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.resolutions = resolutions

#         self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, use_flash)

#         self.linear_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, vocab_size)
#         )
        
#         self.LEVEL_RANGES = [
#             (0, 1),    # Level 1: class token
#             (1, 5),    # Level 2: 2x2 tokens
#             (5, 21),   # Level 3: 4x4 tokens
#             (21, 85),  # Level 4: 8x8 tokens
#             (85, 341)  # Level 5: 16x16 tokens
#         ]
        
#         self.tokenizer = VQMultiModel.load_from_checkpoint("imagenet_256/vqgan_8192_16/last.ckpt", n_embed=2048)
#         self.tokenizer.to("cuda")
#         self.tokenizer.eval()

#         self.difficulty = DifficultyViT.load_from_checkpoint("./imagenet_256/diffvit/last.ckpt", vocab_size=9216, resolutions=[1, 2, 4, 8, 16], hidden_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=128)
#         self.difficulty.to("cuda")
#         self.difficulty.eval()
        
#         # Create adjacency on cuda first
#         T = 341
#         adjacency = torch.full((T, 4), -1, dtype=torch.long)
#         for i in range(T):
#             inds = self.get_subpatch_indices(i)
#             for j, sp_i in enumerate(inds):
#                 adjacency[i, j] = sp_i
        
#         # Register as buffer and move to current device
#         self.register_buffer('adjacency', adjacency, persistent=False)

#     def find_level(self, i: int) -> int:
#         """Returns which hierarchical level (0-5) the token index belongs to."""
#         for level, (start, end) in enumerate(self.LEVEL_RANGES):
#             if start <= i < end:
#                 return level
#         return -1

#     def get_subpatch_indices(self, i: int) -> list:
#         """Returns indices of 4 sub-patches in next level for given token index."""
#         level = self.find_level(i)
#         if level >= 4:
#             return []
#         start = self.LEVEL_RANGES[level][0]
#         next_start = self.LEVEL_RANGES[level + 1][0]
#         offset = i - start
#         return [next_start + (offset * 4) + j for j in range(4)]

#     def forward(self, patches):
#         x = self.token_embedding(patches)

#         b, _, dim = x.shape
#         idx = 0
        
#         # For storing the pos-encoded chunks to be concatenated later
#         pos_encoded_chunks = []

#         for resolution in self.resolutions:
#             num_patches_for_res = resolution * resolution

#             # Extract chunk from x for this resolution and reshape to (b, h, w, dim)
#             chunk = x[:, idx : idx + num_patches_for_res, :]
#             chunk = chunk.reshape(b, resolution, resolution, dim)

#             # Compute sin-cos position embeddings in 2D
#             pe = posemb_sincos_2d(chunk)  # should return shape (b, resolution * resolution, dim)

#             # Flatten back to (b, h*w, dim)
#             chunk = chunk.reshape(b, -1, dim)
#             # Add position embeddings
#             chunk = chunk + pe

#             pos_encoded_chunks.append(chunk)
#             idx += num_patches_for_res

#         # Concatenate all chunks back along the sequence dimension
#         x = torch.cat(pos_encoded_chunks, dim=1)  # (b, sum_of_all_r^2, dim)

#         x = self.transformer(x)
        
#         output = self.linear_head(x)

#         return output
    
#     def ndp_step(self, batch, detail_idx=None):
#         """
#         Next-Detail Prediction (NDP) step using masked-language-model-style loss.

#         Args:
#             batch: Dictionary containing 'tokens' and 'difficulties'
#             detail_idx: Optional tensor of indices to use. If None, random indices are generated.

#         Returns:
#             loss: The computed cross entropy loss
#         """
#         tokens = batch["tokens"]             # (B, T)
#         difficulties = batch["difficulties"] # (B, T)
#         B, T = tokens.shape

#         # 1) Use provided indices or randomly pick an index in 1-4 levels for each sample
#         if detail_idx is None:
#             detail_idx = torch.randint(low=0, high=85, size=(B,), device=tokens.device)

#         # We'll keep a clone of tokens so original isn't overwritten
#         tokens_modified = tokens.clone()

#         # This mask will be True in positions we want to compute the cross-entropy over
#         loss_mask = torch.zeros_like(tokens, dtype=torch.bool)

#         chosen_diffs = difficulties[torch.arange(B), detail_idx]  # shape: (B,)
        
#         causal_mask = difficulties < chosen_diffs.unsqueeze(1)

#         # BUT keep token 0 intact
#         causal_mask[:, 0] = False

#         # Now zero them out in one go
#         tokens_modified[causal_mask] = 0
        
#         sp_inds = self.adjacency[detail_idx]
#         # Create batch indices that match sp_inds shape
#         batch_indices = torch.arange(B, device=sp_inds.device)[:, None].expand(-1, 4)
        
#         # Only modify valid indices (where sp_inds != -1)
#         valid_mask = sp_inds != -1
#         tokens_modified[batch_indices[valid_mask], sp_inds[valid_mask]] = 1
#         loss_mask[batch_indices[valid_mask], sp_inds[valid_mask]] = True
        
#         # Now get the sub-sub-patches (level 2)
#         sp_inds_masked = self.adjacency[sp_inds]  # shape: (B, 4, 4)
#         batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, 4)  # shape: (B, 4, 4)
#         valid_mask = sp_inds_masked != -1
#         tokens_modified[batch_indices_expanded[valid_mask], sp_inds_masked[valid_mask]] = 0

#         # Get sub-sub-sub-patches (level 3)
#         sp_inds_masked_2 = self.adjacency[sp_inds_masked]  # shape: (B, 4, 4, 4)
#         batch_indices_expanded_2 = batch_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, 4)
#         valid_mask_2 = sp_inds_masked_2 != -1
#         tokens_modified[batch_indices_expanded_2[valid_mask_2], sp_inds_masked_2[valid_mask_2]] = 0

#         # Get level 4 patches
#         sp_inds_masked_3 = self.adjacency[sp_inds_masked_2]  # shape: (B, 4, 4, 4, 4)
#         batch_indices_expanded_3 = batch_indices_expanded_2.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
#         valid_mask_3 = sp_inds_masked_3 != -1
#         tokens_modified[batch_indices_expanded_3[valid_mask_3], sp_inds_masked_3[valid_mask_3]] = 0
#         # print("Input", tokens_modified)

#         # 4) Forward pass on the modified tokens
#         logits = self.forward(tokens_modified)  # shape: (B, T, vocab_size)
#         # Create targets with -100 for non-masked positions (ignored by cross_entropy)
#         masked_targets = torch.full_like(tokens, -100)
#         masked_targets[loss_mask] = tokens[loss_mask]      # Fill in actual targets for masked positions
        
#         # print("Predicting", masked_targets)

#         return F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1))
        
#     def training_step(self, batch, batch_idx):
#         # Randomly set 10% of the batch to [2]
#         # Only modify first token in sequence
#         # Generate random values on same device as tokens
#         rand_vals = torch.rand(batch["tokens"].shape[0], device=batch["tokens"].device)
#         batch["tokens"][:, 0] = torch.where(rand_vals < 0.1,
#                                           torch.tensor(2, device=batch["tokens"].device),
#                                           batch["tokens"][:, 0] + 8)
#         loss = self.ndp_step(batch)
#         self.log("train_loss", loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         # Use deterministic indices based on batch size
#         B = batch["tokens"].shape[0]
#         detail_idx = torch.arange(0, B, device=batch["tokens"].device) % 85
#         loss = self.ndp_step(batch, detail_idx)
#         self.log("val_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
#         return optimizer
    
#     def log_images(self, batch, **kwargs):
        
#         tokens = torch.zeros((1, 341), dtype=torch.long, device=self.device)
#         idx = 0
#         tokens[0][0] = batch["tokens"][0][0]

#         def generate_next_detail(tokens, idx):
#             # Mark subpatch indices as masked
#             subpatch_indices = self.get_subpatch_indices(idx)
#             tokens[0, subpatch_indices] = 1

#             # Classifier-free guidance: build an unconditioned version by setting the first token to 2
#             uncond_tokens = tokens.clone()
#             uncond_tokens[0, 0] = 2

#             # Forward pass for conditioned and unconditioned tokens
#             logits = self.forward(tokens.to("cuda"))
#             uncond_logits = self.forward(uncond_tokens.to("cuda"))

#             # Merge conditioned and unconditioned logits
#             guidance_scale = 2.0
#             logits = uncond_logits + guidance_scale * (logits - uncond_logits)

#             # Determine valid token range based on hierarchical level
#             level = self.find_level(idx)
#             if level == 0:  # 2x2 level
#                 valid_range = (1024, 1024 + 2048)
#             elif level == 1:  # 4x4 level
#                 valid_range = (1024 + 2048, 1024 + 2048 + 2048)
#             elif level == 2:  # 8x8 level
#                 valid_range = (1024 + 2048 + 2048, 1024 + 2048 + 2048 + 2048)
#             else:  # 16x16 level
#                 valid_range = (1024 + 2048 + 2048 + 2048, 1024 + 2048 + 2048 + 2048 + 2048)

#             # Mask logits outside the valid range
#             logits_masked = logits[0, subpatch_indices].clone()
#             logits_masked[:, :valid_range[0]] = float('-inf')
#             logits_masked[:, valid_range[1]:] = float('-inf')

#             # Temperature sampling
#             temperature = 0.5
#             logits_masked = logits_masked / temperature

#             # Top-k sampling
#             k = 5
#             probs = torch.softmax(logits_masked, dim=-1)
#             top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
#             top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
#             sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
#             sampled_tokens = top_k_indices[torch.arange(len(sampled_indices)), sampled_indices]

#             # Place the sampled tokens
#             tokens[0, subpatch_indices] = sampled_tokens.to(dtype=tokens.dtype, device=tokens.device)

#             # Compute difficulty for positioned tokens
#             difficulty = self.difficulty.forward(tokens.to("cuda"))

#             # Create mask for non-zero tokens that have all zero subpatches
#             valid_mask = torch.zeros_like(tokens[0], dtype=torch.bool, device=tokens.device)
#             for i in range(len(valid_mask)):
#                 if tokens[0, i] != 0:
#                     sp = self.get_subpatch_indices(i)
#                     if sp:
#                         has_all_zeros = all(tokens[0, j] == 0 for j in sp)
#                         valid_mask[i] = has_all_zeros

#             # Mask difficulty to only consider valid tokens
#             masked_difficulty = torch.where(
#                 valid_mask.to(difficulty.device),
#                 difficulty[0],
#                 torch.tensor(float('-inf'), device=difficulty.device)
#             )

#             # Highest-difficulty valid token index
#             max_idx = torch.argmax(masked_difficulty)
#             return tokens, max_idx.item()

#         output = []
#         for i in range(64):
#             tokens, idx = generate_next_detail(tokens, idx)
            
#             if i in [0, 3, 7, 15, 31, 63]:  # Plot at 1, 4, 8, 16, 32, 64 details
#                 with torch.no_grad():
#                     # Handle final offsets for each level
#                     level1_tokens = torch.where(tokens[0, 1:5] != 0, tokens[0, 1:5] - 1024, tokens[0, 1:5])
#                     level2_tokens = torch.where(tokens[0, 5:21] != 0, tokens[0, 5:21] - (1024 + 2048), tokens[0, 5:21])
#                     level3_tokens = torch.where(tokens[0, 21:85] != 0, tokens[0, 21:85] - (1024 + 2048 + 2048), tokens[0, 21:85])
#                     level4_tokens = torch.where(tokens[0, 85:341] != 0, tokens[0, 85:341] - (1024 + 2048 + 2048 + 2048), tokens[0, 85:341])

#                     # Convert to codebook entries
#                     q1 = self.tokenizer.quantize_1.get_codebook_entry(level4_tokens.to("cuda"), (1, 16, 16, 8))
#                     q2 = self.tokenizer.quantize_2.get_codebook_entry(level3_tokens.to("cuda"), (1, 8, 8, 8))
#                     q3 = self.tokenizer.quantize_3.get_codebook_entry(level2_tokens.to("cuda"), (1, 4, 4, 8))
#                     q4 = self.tokenizer.quantize_4.get_codebook_entry(level1_tokens.to("cuda"), (1, 2, 2, 8))

#                     # Decode image
#                     img = self.tokenizer.decode(q1, q2, q3, q4)
#                     output.append(img)

#         return output
                    