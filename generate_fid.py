import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from data.custom import CustomTrain, CustomTest
from models.vqgan import VQMultiModel
from models.vit import DifficultyViT, NDPViT
from data.utils import custom_collate
import torch

tokenizer = VQMultiModel.load_from_checkpoint("./imagenet_256/vqgan_8192_16/last.ckpt", n_embed=2048)
tokenizer.to("cuda")
tokenizer.eval()

ndpvit = NDPViT.load_from_checkpoint("imagenet_256/ndpvit/epoch=299-step=800940.ckpt", vocab_size=8256, resolutions=[1, 2, 4, 8, 16], hidden_dim=768, depth=12, heads=12, mlp_dim=1536, dim_head=64)
ndpvit.to("cuda")
ndpvit.eval()




# Create output directory if it doesn't exist
os.makedirs("./imagenet_256/samples_fid/", exist_ok=True)

# Boundaries for each resolution level
LEVEL_RANGES = [
    (0, 1),    # Level 1: class token
    (1, 5),    # Level 2: 2x2 tokens
    (5, 21),   # Level 3: 4x4 tokens
    (21, 85),  # Level 4: 8x8 tokens
    (85, 341)  # Level 5: 16x16 tokens
]

def find_level(i: int) -> int:
    for level, (start, end) in enumerate(LEVEL_RANGES):
        if start <= i < end:
            return level
    return -1

def get_subpatch_indices(i: int) -> list:
    level = find_level(i)
    if level >= 4:
        return []
    start = LEVEL_RANGES[level][0]
    next_start = LEVEL_RANGES[level + 1][0]
    offset = i - start
    return [next_start + (offset * 4) + j for j in range(4)]

def generate_next_detail(tokens, idx):
    subpatch_indices = get_subpatch_indices(idx)
    tokens[:, subpatch_indices] = 1
    uncond_tokens = tokens.clone()
    uncond_tokens[:, 0] = 1000
    
    with torch.no_grad():
        logits = ndpvit.forward(tokens.to("cuda"))
        uncond_logits = ndpvit.forward(uncond_tokens.to("cuda"))

    guidance_scale = 1.5
    logits = uncond_logits + guidance_scale * (logits - uncond_logits)

    level = find_level(idx)
    if level == 0:
        valid_range = (64, 64 + 2048)
    elif level == 1:
        valid_range = (64 + 2048, 64 + 2048 + 2048)
    elif level == 2:
        valid_range = (64 + 2048 + 2048, 64 + 2048 + 2048 + 2048)
    else:
        valid_range = (64 + 2048 + 2048 + 2048, 64 + 2048 + 2048 + 2048 + 2048)

    logits_masked = logits[0, subpatch_indices].clone()
    logits_masked[:, :valid_range[0]] = float('-inf')
    logits_masked[:, valid_range[1]:] = float('-inf')

    temperature = 0.95
    logits_masked = logits_masked / temperature

    k = 20
    probs = torch.softmax(logits_masked, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
    sampled_tokens = top_k_indices[torch.arange(len(sampled_indices)), sampled_indices]

    tokens[:, subpatch_indices] = sampled_tokens.to(dtype=tokens.dtype, device=tokens.device)

    return tokens, idx + 1

def generate_image(class_idx):
    tokens = torch.zeros(2, 341, dtype=torch.int)
    tokens[:, 0] = class_idx
    idx = 0
    
    for _ in range(64):
        tokens, idx = generate_next_detail(tokens, idx)
    
    with torch.no_grad():
        level1_tokens = torch.where(tokens[:, 1:5] != 0, tokens[:, 1:5] - 64, tokens[:, 1:5])
        level2_tokens = torch.where(tokens[:, 5:21] != 0, tokens[:, 5:21] - (64 + 2048), tokens[:, 5:21])
        level3_tokens = torch.where(tokens[:, 21:85] != 0, tokens[:, 21:85] - (64 + 2048 + 2048), tokens[:, 21:85])
        level4_tokens = torch.where(tokens[:, 85:341] != 0, tokens[:, 85:341] - (64 + 2048 + 2048 + 2048), tokens[:, 85:341])

        q1 = tokenizer.quantize_1.get_codebook_entry(level4_tokens.to("cuda"), (2, 16, 16, 8))
        q2 = tokenizer.quantize_2.get_codebook_entry(level3_tokens.to("cuda"), (2, 8, 8, 8))
        q3 = tokenizer.quantize_3.get_codebook_entry(level2_tokens.to("cuda"), (2, 4, 4, 8))
        q4 = tokenizer.quantize_4.get_codebook_entry(level1_tokens.to("cuda"), (2, 2, 2, 8))
        
        img = tokenizer.decode(q1, q2, q3, q4)
        return img

# Generate 50 samples for each class
for class_idx in tqdm(range(1000), desc="Generating samples"):
    for sample_idx in range(25):
        img = generate_image(class_idx)
        sample_idx = sample_idx * 2
        save_image(img[0], f"./imagenet_256/samples_fid/class_{class_idx:04d}_sample_{sample_idx:02d}.png", normalize=True)
        sample_idx += 1
        save_image(img[1], f"./imagenet_256/samples_fid/class_{class_idx:04d}_sample_{sample_idx:02d}_uncond.png", normalize=True)
