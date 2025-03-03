import math
import inspect
from dataclasses import dataclass
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.vqgan import IBQSharedModel
from tqdm import tqdm

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    context_length: int = 1024
    content_vocab_size: int = 50304
    position_vocab_size: int = 1024
    n_classes: int = 1000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.999)
    tokenizer: IBQSharedModel = None

class GPT(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        assert config.content_vocab_size is not None
        assert config.context_length is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wce = nn.Embedding(config.n_classes, config.n_embd),
            wte = nn.Embedding(config.content_vocab_size + 1, config.n_embd),
            wpe = nn.Embedding(config.position_vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=False),
        ))
        self.content_head = nn.Linear(config.n_embd, self.config.content_vocab_size + 1, bias=False)
        self.position_head = nn.Linear(config.n_embd, config.position_vocab_size, bias=False)
        self.tokenizer = config.tokenizer.to(self.device)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, content_tokens, position_tokens, content_targets=None, position_targets=None):
        device = content_tokens.device
        b, t = content_tokens.size()
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"
        
        # print('Content tokens', content_tokens)
        # print('Position tokens', position_tokens)
        # if content_targets is not None:
        #     print('Content targets', content_targets)
        #     print('Position targets', position_targets)
        # forward the GPT model itself
        if(content_tokens.shape[1] > 1):
            class_token = content_tokens[:, :1]
            content_tokens = content_tokens [:, 1:]
            class_emb = self.transformer.wce(class_token)
            tok_emb = self.transformer.wte(content_tokens) # token embeddings of shape (b, t, n_embd)
            tok_emb = torch.cat((class_emb, tok_emb), dim=1)
        else:
            class_emb = self.transformer.wce(content_tokens)
            tok_emb = class_emb
       
        pos_emb = self.transformer.wpe(position_tokens) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if content_targets is not None and position_targets is not None:
            # if we are given some desired targets also calculate the loss
            content_logits = self.content_head(x)
            position_logits = self.position_head(x)
            content_loss = F.cross_entropy(content_logits.view(-1, content_logits.size(-1)), content_targets.view(-1), ignore_index=-1)
            position_loss = F.cross_entropy(position_logits.view(-1, position_logits.size(-1)), position_targets.view(-1), ignore_index=-1)
            loss = content_loss + position_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            content_logits = self.content_head(x)
            position_logits = self.position_head(x)
            loss = None

        return content_logits, position_logits, loss
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        
        _, _, _, _, _, ((_, _, ind1), (_, _, ind2), (_, _, ind3), (_, _, ind4)) = self.tokenizer.encode(images)
        
        ind1 = torch.reshape(ind1, (images.size(0), -1)) + self.config.n_classes
        ind2 = torch.reshape(ind2, (images.size(0), -1)) + self.config.n_classes
        ind3 = torch.reshape(ind3, (images.size(0), -1)) + self.config.n_classes
        ind4 = torch.reshape(ind4, (images.size(0), -1)) + self.config.n_classes
        
        content_tokens = torch.cat((labels.unsqueeze(1), ind4, ind3, ind2, ind1), dim=1)
        position_tokens = torch.arange(content_tokens.size(1)).unsqueeze(0).expand(content_tokens.size(0), -1).to(self.device)

        position_targets = position_tokens.clone() + 1
        content_targets = torch.cat((content_tokens[:, 1:], torch.full((content_tokens.size(0), 1), self.config.content_vocab_size + 1, device=content_tokens.device)), dim=1)
        
        content_logits, position_logits, loss = self.forward(content_tokens, position_tokens, content_targets, position_targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        
        _, _, _, _, _, ((_, _, ind1), (_, _, ind2), (_, _, ind3), (_, _, ind4)) = self.tokenizer.encode(images)
        
        ind1 = torch.reshape(ind1, (images.size(0), -1))
        ind2 = torch.reshape(ind2, (images.size(0), -1))
        ind3 = torch.reshape(ind3, (images.size(0), -1)) 
        ind4 = torch.reshape(ind4, (images.size(0), -1))
        
        content_tokens = torch.cat((labels.unsqueeze(1), ind4, ind3, ind2, ind1), dim=1)
        position_tokens = torch.arange(content_tokens.size(1)).unsqueeze(0).expand(content_tokens.size(0), -1).to(self.device)

        position_targets = position_tokens.clone() + 1
        content_targets = torch.cat((content_tokens[:, 1:], torch.full((content_tokens.size(0), 1), self.config.content_vocab_size + 1, device=content_tokens.device)), dim=1)
        
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
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, labels, positions, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        content_tokens = labels.to(self.device)
        position_tokens = positions.to(self.device)
        
        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            content_idx_cond = content_tokens if content_tokens.size(1) <= self.config.context_length else content_tokens[:, -self.config.context_length:]
            position_idx_cond = position_tokens if position_tokens.size(1) <= self.config.context_length else position_tokens[:, -self.config.context_length:]
            # forward the model to get the logits for the index in the sequence
            content_logits, position_logits, _ = self(content_idx_cond, position_idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            content_logits = content_logits[:, -1, :] / temperature
            position_logits = position_logits[:, -1, :] / temperature
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
            
        return content_tokens, position_tokens