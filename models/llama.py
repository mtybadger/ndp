import math
import inspect
from dataclasses import dataclass
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.vqgan import IBQSharedModel
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from tqdm import tqdm
from torch.nn.functional import scaled_dot_product_attention
from torch import Tensor
from modules.muon import Muon
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def rotate(x, freqs_cis, position_tokens):
    selected_freqs = freqs_cis[position_tokens]
    return apply_rotary_emb(x, selected_freqs)


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_head, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_head, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.kv_cache = None
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis, position_tokens, use_cache=False, debug=False):
        with torch.amp.autocast(device_type=x.device.type):
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            q, k = norm(q), norm(k)
            
            # print('Position tokens', position_tokens, position_tokens.shape, position_tokens.dtype)
            
            if position_tokens.shape[1] > T:
                q = rotate(q, freqs_cis, position_tokens[:, 1:])
                k = rotate(k, freqs_cis, position_tokens[:, :-1])
            else:
                q = rotate(q, freqs_cis, position_tokens.roll(-1, dims=1))
                k = rotate(k, freqs_cis, position_tokens)
                
            if use_cache and not self.training:
                k_cache = self.kv_cache.k_cache
                v_cache = self.kv_cache.v_cache
                k = k[:, -1:, :, :]
                v = v[:, -1:, :, :]
                y = flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=T-1, causal=True)
            else:
                # if q.dtype in (torch.float16, torch.bfloat16):
                #     print('Using flash attention')
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
                # else:
                #     y = scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), dropout_p=self.dropout if self.training else 0.0, is_causal=True).transpose(1,2)
            
            y = y.view(B, T, C)
            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.silu = nn.SiLU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)
        self.drop_path = DropPath(config.dropout) if config.dropout > 0. else nn.Identity()

    def forward(self, x, freqs_cis, position_tokens, use_cache=False, debug=False):
        x = x + self.drop_path(self.attn(self.ln_1(x), freqs_cis, position_tokens, use_cache=use_cache, debug=debug))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

@dataclass
class LlamaConfig:
    context_length: int = 341
    content_vocab_size: int = 16384
    position_vocab_size: int = 341
    n_classes: int = 1000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.95)
    norm_eps: float = 1e-5
    ndp: bool = False
    image_resolution: int = 64

class Llama(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super().__init__()
        assert config.content_vocab_size is not None
        assert config.context_length is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wce = nn.Embedding(config.n_classes + 1, config.n_embd),
            wte = nn.Embedding(config.content_vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.content_head = nn.Linear(config.n_embd, self.config.content_vocab_size, bias=False)
        self.position_head = nn.Linear(config.n_embd, config.position_vocab_size, bias=False)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per Llama-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        self.tokenizer = tokenizer.to(self.device)
        self.tokenizer.eval()
        self.tokenizer.half()
        for param in self.tokenizer.parameters():
            param.requires_grad = False
            
        pos_emb = RotaryEmbedding(
            dim = 16,
            freqs_for = 'pixel',
            max_freq = 256
        )
        
        freqs0 = pos_emb.get_axial_freqs(5, 1, 1)[0].view(-1, 48)
        freqs1 = pos_emb.get_axial_freqs(5, 2, 2)[1].view(-1, 48)    
        freqs2 = pos_emb.get_axial_freqs(5, 4, 4)[2].view(-1, 48)
        freqs3 = pos_emb.get_axial_freqs(5, 8, 8)[3].view(-1, 48)
        freqs4 = pos_emb.get_axial_freqs(5, 16, 16)[4].view(-1, 48)


            
        self.freqs_cis = torch.cat((freqs0, freqs1, freqs2, freqs3, freqs4), dim=0)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self.transformer = torch.compile(self.transformer)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def init_kv_cache(self, batch_size, seq_length):
        for block in self.transformer.h:
            block.attn.kv_cache = KVCache(batch_size, seq_length, self.config.n_head, self.config.n_embd // self.config.n_head, self.dtype, self.device)

    def forward(self, content_tokens, position_tokens, content_targets=None, position_targets=None, use_cache=False):
        device = content_tokens.device
        b, t = content_tokens.size()
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"
        
        # print('Content tokens', content_tokens, content_tokens.shape, content_tokens.dtype)
        # print('Position tokens', position_tokens, position_tokens.shape, position_tokens.dtype)
        # if content_targets is not None:
        #     print('Content targets', content_targets, content_targets.shape, content_targets.dtype)
        # if position_targets is not None:
        #     print('Position targets', position_targets, position_targets.shape, position_targets.dtype)
        
        self.freqs_cis = self.freqs_cis.to(position_tokens.device)
        
        
        # forward the Llama model itself
        if(content_tokens.shape[1] > 1):
            class_token = content_tokens[:, :1]
            content_tokens = content_tokens [:, 1:]
            class_emb = self.transformer.wce(class_token)
            tok_emb = self.transformer.wte(content_tokens) # token embeddings of shape (b, t, n_embd)
            tok_emb = torch.cat((class_emb, tok_emb), dim=1)
        else:
            class_emb = self.transformer.wce(content_tokens)
            tok_emb = class_emb
            
        x = self.transformer.drop(tok_emb)
        for (i, block) in enumerate(self.transformer.h):
            x = block(x, freqs_cis=self.freqs_cis, position_tokens=position_tokens, use_cache=use_cache, debug=i == 0)
        x = self.transformer.ln_f(x)

        if content_targets is not None and position_targets is not None:
            # if we are given some desired targets also calculate the loss
            content_logits = self.content_head(x)
            position_logits = self.position_head(x)
            content_loss = F.cross_entropy(content_logits.view(-1, content_logits.size(-1)), content_targets.view(-1), ignore_index=-100)
            position_loss = F.cross_entropy(position_logits.view(-1, position_logits.size(-1)), position_targets.view(-1), ignore_index=-100)
            loss = content_loss + position_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            content_logits = self.content_head(x)
            position_logits = self.position_head(x)
            loss = None

        return content_logits, position_logits, loss
    
    def calculate_patchwise_loss(self, original, reconstructed, patch_size):
        """Calculate MSE loss for each patch size"""
        losses = {}
        batch, channels, height, width = original.shape
        
        # Calculate number of patches
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        # Reshape tensors to extract all patches at once
        # [B, C, H, W] -> [B, C, h_patches, patch_size, w_patches, patch_size]
        orig_patches = original.view(batch, channels, h_patches, patch_size, w_patches, patch_size)
        recon_patches = reconstructed.view(batch, channels, h_patches, patch_size, w_patches, patch_size)
        
        # Permute and reshape to get all patches: [B, h_patches*w_patches, C, patch_size, patch_size]
        orig_patches = orig_patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, h_patches*w_patches, -1)
        recon_patches = recon_patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, h_patches*w_patches, -1)
        
        # Calculate MSE for all patches simultaneously
        patch_losses = torch.sqrt(((orig_patches - recon_patches) ** 2).sum(dim=2))
        
        return patch_losses
    
    def tokenize(self, images, labels, training=False):
        with torch.amp.autocast(device_type=self.device.type):
            with torch.inference_mode():
                q1, q2, q3, q4, _, ((_, _, ind1), (_, _, ind2), (_, _, ind3), (_, _, ind4)) = self.tokenizer.encode(images)
                
        ind1 = torch.reshape(ind1, (images.size(0), -1))
        ind2 = torch.reshape(ind2, (images.size(0), -1))
        ind3 = torch.reshape(ind3, (images.size(0), -1))
        ind4 = torch.reshape(ind4, (images.size(0), -1))
        class_token = labels.unsqueeze(1)
        if torch.rand(1) < 0.1:
            class_token = torch.full((labels.size(0), 1), self.config.n_classes, device=labels.device)

        if self.config.ndp:
            with torch.amp.autocast(device_type=self.device.type):
                with torch.inference_mode():
                    z1, z2, z3, z4 = self.tokenizer.get_zero_tokens(images.size(0), self.tokenizer.embed_dim, self.device)
                    
                    # Create batched inputs for decoder
                    batch_size = images.size(0)
                    q1_batch = torch.cat([z1, z1, z1, q1], dim=0) 
                    q2_batch = torch.cat([z2, z2, q2, q2], dim=0)
                    q3_batch = torch.cat([z3, q3, q3, q3], dim=0)
                    q4_batch = torch.cat([q4, q4, q4, q4], dim=0)
                    
                    # Single decode call
                    reconstructed = self.tokenizer.decode(q1_batch, q2_batch, q3_batch, q4_batch)
                    
                    # Split reconstructed batch back into individual outputs
                    recon_4, recon_3, recon_2, recon_1 = torch.split(reconstructed, batch_size)
                    
                    # Calculate losses
                    losses_4 = self.calculate_patchwise_loss(images, recon_4, self.config.image_resolution // 2)
                    losses_3 = self.calculate_patchwise_loss(images, recon_3, self.config.image_resolution // 4) 
                    losses_2 = self.calculate_patchwise_loss(images, recon_2, self.config.image_resolution // 8)
                    losses_1 = self.calculate_patchwise_loss(images, recon_1, self.config.image_resolution // 16)
            all_losses = torch.cat((torch.full((losses_4.size(0), 1), float('inf'), device=self.device, dtype=losses_4.dtype), losses_4, losses_3, losses_2, losses_1), dim=1)
            
            sorted_indices = torch.argsort(all_losses, dim=1, descending=True)
            content_tokens = torch.cat((class_token, ind4, ind3, ind2, ind1), dim=1)
            position_tokens = torch.arange(content_tokens.size(1)).unsqueeze(0).expand(content_tokens.size(0), -1).to(self.device)
            
            content_tokens = torch.gather(content_tokens, dim=1, index=sorted_indices)
            position_tokens = torch.gather(position_tokens, dim=1, index=sorted_indices)
            position_targets = torch.cat((position_tokens[:, 2:], torch.full((position_tokens.size(0), 2), -100, device=position_tokens.device, dtype=torch.long)), dim=1)

        else:
            content_tokens = torch.cat((class_token, ind4, ind3, ind2, ind1), dim=1)
            position_tokens = torch.arange(content_tokens.size(1)).unsqueeze(0).expand(content_tokens.size(0), -1).to(self.device, dtype=torch.long)
            position_targets = torch.cat((position_tokens[:, 1:], torch.full((position_tokens.size(0), 1), -100, device=position_tokens.device, dtype=torch.long)), dim=1)
            
        content_targets = torch.cat((content_tokens[:, 1:], torch.full((content_tokens.size(0), 1), -100, device=content_tokens.device, dtype=torch.long)), dim=1)
        
        # print('Content tokens', content_tokens)
        # print('Position tokens', position_tokens)
        # print('Content targets', content_targets)
        # print('Position targets', position_targets)
        return content_tokens, position_tokens, content_targets, position_targets
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        content_tokens, position_tokens, content_targets, position_targets = self.tokenize(images, labels, training=True)
        # print('Content targets', content_targets)
        content_logits, position_logits, loss = self.forward(content_tokens, position_tokens, content_targets, position_targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        content_tokens, position_tokens, content_targets, position_targets = self.tokenize(images, labels, training=False)
        
        content_logits, position_logits, loss = self.forward(content_tokens, position_tokens, content_targets, position_targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # optim_groups = [
        #     {'params': decay_params, 'weight_decay': self.config.weight_decay},
        #     {'params': nodecay_params, 'weight_decay': 0.0}
        # ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and torch.cuda.is_available()
        # extra_args = dict(fused=True) if use_fused else dict()
        optimizer = Muon(lr=self.learning_rate, wd=self.config.weight_decay, muon_params=decay_params, adamw_params=nodecay_params, adamw_betas=self.config.betas, adamw_eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.001,
            total_iters=300
        )
        # print(f"using fused AdamW: {use_fused}")

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    @torch.no_grad()
    def generate(self, labels, positions, max_new_tokens, temperature=1.0, top_k=None, cfg_scale=1.0, kv_caching=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        content_tokens = labels.to(self.device)
        position_tokens = positions.to(self.device)
        
        if kv_caching:
            if cfg_scale != 1.0:
                self.init_kv_cache(content_tokens.size(0) * 2, 341)
            else:
                self.init_kv_cache(content_tokens.size(0), 341)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            content_idx_cond = content_tokens if content_tokens.size(1) <= self.config.context_length else content_tokens[:, -self.config.context_length:]
            position_idx_cond = position_tokens if position_tokens.size(1) <= self.config.context_length else position_tokens[:, -self.config.context_length:]
            
            content_idx_uncond = content_tokens.clone()
            content_idx_uncond[:, 0] = self.config.n_classes

            # forward the model to get the logits for the index in the sequence
            if cfg_scale == 1.0:
                content_logits, position_logits, _ = self.forward(content_idx_cond, position_idx_cond, use_cache=kv_caching)
            else:
                content_idx = torch.cat((content_idx_cond, content_idx_uncond), dim=0)
                position_idx = torch.cat((position_idx_cond, position_idx_cond), dim=0)
                logits, position_logits, _ = self.forward(content_idx, position_idx, use_cache=kv_caching)
                content_logits = logits[:content_idx_cond.size(0)]
                position_logits = position_logits[:position_idx_cond.size(0)]
                content_logits_uncond = logits[content_idx_cond.size(0):]
                content_logits = content_logits_uncond + cfg_scale * (content_logits - content_logits_uncond)
            # pluck the logits at the final step and scale by desired temperature
            content_logits = content_logits[:, -1, :] / temperature
            position_logits = position_logits[:, -1, :] / temperature
            
            # ---- Mask out already used position tokens ----
            # For each batch element, set logits for tokens already in position_tokens to -inf.
            batch_size = position_logits.size(0)
            vocab_size = position_logits.size(-1)
            # Create a mask (True means token is allowed).
            mask = torch.ones((batch_size, vocab_size), device=position_logits.device, dtype=torch.bool)
            for i in range(batch_size):
                # Get the unique tokens used so far in the positions sequence for this batch.
                used_tokens = position_tokens[i].unique()
                # Only mask tokens if we haven't used all possible positions yet
                if len(used_tokens) < vocab_size:
                    mask[i, used_tokens] = False
            mask[:, 0] = True
            # Apply the mask: disallowed tokens get a logit of -infinity.
            position_logits = position_logits.masked_fill(~mask, -float('Inf'))
            # --------------------------------------------------
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(content_logits, min(top_k, content_logits.size(-1)))
                content_logits[content_logits < v[:, [-1]]] = -float('Inf')
            if top_k is not None:
                v, _ = torch.topk(position_logits, min(top_k, position_logits.size(-1)))
                position_logits[position_logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            content_probs = F.softmax(content_logits, dim=-1)
            position_probs = F.softmax(position_logits, dim=-1)
            
            # sample from the distribution
            content_idx_next = torch.multinomial(content_probs, num_samples=1)
            position_idx_next = torch.multinomial(position_probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            content_tokens = torch.cat((content_tokens, content_idx_next), dim=1)
            position_tokens = torch.cat((position_tokens, position_idx_next), dim=1)
            
            # print('Content tokens', content_tokens)
            # print('Position tokens', position_tokens)

        return content_tokens, position_tokens