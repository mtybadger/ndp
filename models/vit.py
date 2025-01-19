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

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# blocks = torch.cat([
#     torch.zeros(1),  # 1x1 resolution
#     torch.ones(4),  # 2x2 resolution 
#     torch.full((16,), 2),  # 4x4 resolution
#     torch.full((64,), 3),  # 8x8 resolution
#     torch.full((512,), 4)  # 16x16 resolution
# ], dim=0).to("cuda")
# # helpers
# def causal(b, h, q_idx, kv_idx):
#     # Hardcoded boundaries based on resolutions [1,2,4,8,16]
#     # boundaries[i+1] = boundaries[i] + res[i]^2
#     return blocks[q_idx] >= blocks[kv_idx]

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

        out = F.scaled_dot_product_attention(q, k, v)
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
    def __init__(self, *, vocab_size, resolutions, hidden_dim, depth, heads, mlp_dim, dim_head = 64, use_flash = True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.resolutions = resolutions

        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, use_flash)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.LEVEL_RANGES = [
            (0, 1),    # Level 1: class token
            (1, 5),    # Level 2: 2x2 tokens
            (5, 21),   # Level 3: 4x4 tokens
            (21, 85),  # Level 4: 8x8 tokens
            (85, 341)  # Level 5: 16x16 tokens
        ]
        
        self.tokenizer = VQMultiModel.load_from_checkpoint("./imagenet/model/last.ckpt")
        self.tokenizer.to("cuda")
        self.tokenizer.eval()

        self.difficulty = DifficultyViT.load_from_checkpoint("./imagenet/diffvit/last.ckpt", vocab_size=5120, resolutions=[1, 2, 4, 8, 16], hidden_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=128)
        self.difficulty.to("cuda")
        self.difficulty.eval()
        
        # Create adjacency on CPU first
        T = 341
        adjacency = torch.full((T, 4), -1, dtype=torch.long)
        for i in range(T):
            inds = self.get_subpatch_indices(i)
            for j, sp_i in enumerate(inds):
                adjacency[i, j] = sp_i
        
        # Register as buffer and move to current device
        self.register_buffer('adjacency', adjacency, persistent=False)

    def find_level(self, i: int) -> int:
        """Returns which hierarchical level (0-5) the token index belongs to."""
        for level, (start, end) in enumerate(self.LEVEL_RANGES):
            if start <= i < end:
                return level
        return -1

    def get_subpatch_indices(self, i: int) -> list:
        """Returns indices of 4 sub-patches in next level for given token index."""
        level = self.find_level(i)
        if level >= 4:
            return []
        start = self.LEVEL_RANGES[level][0]
        next_start = self.LEVEL_RANGES[level + 1][0]
        offset = i - start
        return [next_start + (offset * 4) + j for j in range(4)]

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
        
        output = self.linear_head(x)

        return output
    
    def ndp_step(self, batch, detail_idx=None):
        """
        Next-Detail Prediction (NDP) step using masked-language-model-style loss.

        Args:
            batch: Dictionary containing 'tokens' and 'difficulties'
            detail_idx: Optional tensor of indices to use. If None, random indices are generated.

        Returns:
            loss: The computed cross entropy loss
        """
        tokens = batch["tokens"]             # (B, T)
        difficulties = batch["difficulties"] # (B, T)
        B, T = tokens.shape

        # 1) Use provided indices or randomly pick an index in 1-4 levels for each sample
        if detail_idx is None:
            detail_idx = torch.randint(low=0, high=85, size=(B,), device=tokens.device)

        # We'll keep a clone of tokens so original isn't overwritten
        tokens_modified = tokens.clone()

        # This mask will be True in positions we want to compute the cross-entropy over
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)

        chosen_diffs = difficulties[torch.arange(B), detail_idx]  # shape: (B,)
        
        causal_mask = difficulties < chosen_diffs.unsqueeze(1)

        # BUT keep token 0 intact
        causal_mask[:, 0] = False

        # Now zero them out in one go
        tokens_modified[causal_mask] = 0
        
        sp_inds = self.adjacency[detail_idx]
        # Create batch indices that match sp_inds shape
        batch_indices = torch.arange(B, device=sp_inds.device)[:, None].expand(-1, 4)
        
        # Only modify valid indices (where sp_inds != -1)
        valid_mask = sp_inds != -1
        tokens_modified[batch_indices[valid_mask], sp_inds[valid_mask]] = 1
        loss_mask[batch_indices[valid_mask], sp_inds[valid_mask]] = True
        
        # Now get the sub-sub-patches (level 2)
        sp_inds_masked = self.adjacency[sp_inds]  # shape: (B, 4, 4)
        batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, 4)  # shape: (B, 4, 4)
        valid_mask = sp_inds_masked != -1
        tokens_modified[batch_indices_expanded[valid_mask], sp_inds_masked[valid_mask]] = 0

        # Get sub-sub-sub-patches (level 3)
        sp_inds_masked_2 = self.adjacency[sp_inds_masked]  # shape: (B, 4, 4, 4)
        batch_indices_expanded_2 = batch_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, 4)
        valid_mask_2 = sp_inds_masked_2 != -1
        tokens_modified[batch_indices_expanded_2[valid_mask_2], sp_inds_masked_2[valid_mask_2]] = 0

        # Get level 4 patches
        sp_inds_masked_3 = self.adjacency[sp_inds_masked_2]  # shape: (B, 4, 4, 4, 4)
        batch_indices_expanded_3 = batch_indices_expanded_2.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        valid_mask_3 = sp_inds_masked_3 != -1
        tokens_modified[batch_indices_expanded_3[valid_mask_3], sp_inds_masked_3[valid_mask_3]] = 0
        # print("Input", tokens_modified)

        # 4) Forward pass on the modified tokens
        logits = self.forward(tokens_modified)  # shape: (B, T, vocab_size)
        # Create targets with -100 for non-masked positions (ignored by cross_entropy)
        masked_targets = torch.full_like(tokens, -100)
        masked_targets[loss_mask] = tokens[loss_mask]      # Fill in actual targets for masked positions
        
        # print("Predicting", masked_targets)

        return F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1))
        
    def training_step(self, batch, batch_idx):
        # Randomly set 10% of the batch to [2]
        # Only modify first token in sequence
        # Generate random values on same device as tokens
        rand_vals = torch.rand(batch["tokens"].shape[0], device=batch["tokens"].device)
        batch["tokens"][:, 0] = torch.where(rand_vals < 0.1,
                                          torch.tensor(2, device=batch["tokens"].device),
                                          batch["tokens"][:, 0] + 8)
        loss = self.ndp_step(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Use deterministic indices based on batch size
        B = batch["tokens"].shape[0]
        detail_idx = torch.arange(0, B, device=batch["tokens"].device) % 85
        loss = self.ndp_step(batch, detail_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    def log_images(self, batch, **kwargs):
        
        tokens = torch.zeros((1, 341), dtype=torch.long, device=self.device)
        idx = 0
        tokens[0][0] = batch["tokens"][0][0]
                
        def generate_next_detail(tokens, idx):
            subpatch_indices = self.get_subpatch_indices(idx)
            tokens[0, subpatch_indices] = 1
            logits = self.forward(tokens.to("cuda"))
            # Get level of current index to determine valid token range
            level = self.find_level(idx)
            if level == 0:  # 2x2 level
                valid_range = (1024, 1024+1024)
            elif level == 1:  # 4x4 level 
                valid_range = (1024+1024, 1024+1024+1024)
            elif level == 2:  # 8x8 level
                valid_range = (1024+1024+1024, 1024+1024+1024+1024)
            else:  # 16x16 level
                valid_range = (1024+1024+1024+1024, 1024+1024+1024+1024+1024)
            # Mask logits outside valid range
            logits_masked = logits[0, subpatch_indices].clone()
            logits_masked[:, :valid_range[0]] = float('-inf')
            logits_masked[:, valid_range[1]:] = float('-inf')
            
            # Get top k values and indices
            k = 5
            top_k_values, top_k_indices = torch.topk(logits_masked, k, dim=-1)
            
            # Sample from top k using softmax probabilities
            top_k_probs = torch.softmax(top_k_values, dim=-1)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            
            tokens[0, subpatch_indices] = sampled_tokens.to(dtype=tokens.dtype, device=tokens.device)
            difficulty = self.difficulty.forward(tokens.to("cuda"))
            # Create mask for non-zero tokens that have all zero subpatches
            valid_mask = torch.zeros_like(tokens[0], dtype=torch.bool, device=tokens.device)
            for i in range(len(valid_mask)):
                # Check if current token is non-zero
                if tokens[0, i] != 0:
                    subpatches = self.get_subpatch_indices(i)
                    if subpatches:  # If token has subpatches
                        # Check if all subpatches are zero
                        has_all_zeros = all(tokens[0, j] == 0 for j in subpatches)
                        valid_mask[i] = has_all_zeros
            
            # Mask difficulty to only consider valid tokens
            masked_difficulty = torch.where(valid_mask.to(difficulty.device), difficulty[0], torch.tensor(float('-inf'), device=difficulty.device))
            
            # Get index of highest valid difficulty
            max_idx = torch.argmax(masked_difficulty)
            return tokens, max_idx.item()

        output = []
        for i in range(85):
            tokens, idx = generate_next_detail(tokens, idx)
            
            if i % 20 == 0 and i <= 85:  # Plot every 2 iterations
                # Convert tokens to image
                with torch.no_grad():
                    # Get indices for each resolution level
                    # Only subtract offsets for non-zero tokens
                    level1_tokens = torch.where(tokens[0, 1:5] != 0, tokens[0, 1:5] - 1024, tokens[0, 1:5])  # 2x2
                    level2_tokens = torch.where(tokens[0, 5:21] != 0, tokens[0, 5:21] - (1024 + 1024), tokens[0, 5:21])  # 4x4
                    level3_tokens = torch.where(tokens[0, 21:85] != 0, tokens[0, 21:85] - (1024 + 1024 + 1024), tokens[0, 21:85])  # 8x8
                    level4_tokens = torch.where(tokens[0, 85:341] != 0, tokens[0, 85:341] - (1024 + 1024 + 1024 + 1024), tokens[0, 85:341])  # 16x16
                
                    
                    # Convert to codebook entries
                    q1 = self.tokenizer.quantize_1.get_codebook_entry(level4_tokens.to("cuda"), (1, 16, 16, 8))
                    q2 = self.tokenizer.quantize_2.get_codebook_entry(level3_tokens.to("cuda"), (1, 8, 8, 8))
                    q3 = self.tokenizer.quantize_3.get_codebook_entry(level2_tokens.to("cuda"), (1, 4, 4, 8))
                    q4 = self.tokenizer.quantize_4.get_codebook_entry(level1_tokens.to("cuda"), (1, 2, 2, 8))
                    
                    # Decode image
                    img = self.tokenizer.decode(q1, q2, q3, q4)
                    output.append(img)
                    
        return output
                    
                    
