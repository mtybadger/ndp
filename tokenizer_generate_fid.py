from models.vqgan import IBQSharedModel
from datasets import load_dataset
import os
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Create output directories
os.makedirs("imagenet_64/samples_ibq/level_1", exist_ok=True)
os.makedirs("imagenet_64/samples_ibq/level_2", exist_ok=True)
os.makedirs("imagenet_64/samples_ibq/level_3", exist_ok=True)
os.makedirs("imagenet_64/samples_ibq/level_4", exist_ok=True)

tokenizer = IBQSharedModel(ckpt_path="imagenet_64/vqgan_64_16384_16_ibq/epoch=49-step=250300.ckpt", n_embed=16384, embed_dim=256)
tokenizer.eval()
tokenizer.to("cuda")
for param in tokenizer.parameters():
    param.requires_grad = False
    
tokenizer = torch.compile(tokenizer)

dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")["validation"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def tensor_to_pil(tensor):
    x = tensor.squeeze(0)  # Remove batch dimension
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)  # Denormalize and convert to uint8
    x = x.permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
    x = x.cpu().numpy()
    return Image.fromarray(x)

# Process the dataset in batches
batch_size = 128
all_x1 = []
all_x2 = []
all_x3 = []
all_x4 = []

for idx in tqdm(range(0, len(dataset), batch_size)):
    # Get batch of images
    batch_imgs = []
    for i in range(batch_size):
        if idx + i < len(dataset):
            img = dataset[idx + i]["image"]
            img_tensor = transform(img).unsqueeze(0)
            batch_imgs.append(img_tensor)
    
    # Stack tensors into a batch
    batch_tensor = torch.cat(batch_imgs, dim=0).to("cuda")
    
    # Encode batch
    with torch.no_grad():
        _, _, _, _, _, ((_, _, ind1), (_, _, ind2), (_, _, ind3), (_, _, ind4)) = tokenizer.encode(batch_tensor)
    
    # Get codebook entries
    q1 = tokenizer.quantize.get_codebook_entry(ind1, shape=(batch_tensor.size(0), 16, 16, 256))
    q2 = tokenizer.quantize.get_codebook_entry(ind2, shape=(batch_tensor.size(0), 8, 8, 256))
    q3 = tokenizer.quantize.get_codebook_entry(ind3, shape=(batch_tensor.size(0), 4, 4, 256))
    q4 = tokenizer.quantize.get_codebook_entry(ind4, shape=(batch_tensor.size(0), 2, 2, 256))
    
    # Get zero tokens
    z1, z2, z3, z4 = tokenizer.get_zero_tokens(batch_tensor.size(0), tokenizer.embed_dim, batch_tensor.device)
    
    with torch.no_grad():
        # Generate reconstructions for each level
        x_1 = tokenizer.decode(z1, z2, z3, q4)
        x_2 = tokenizer.decode(z1, z2, q3, q4)
        x_3 = tokenizer.decode(z1, q2, q3, q4)
        x_4 = tokenizer.decode(q1, q2, q3, q4)
        
        # Store reconstructions
        all_x1.extend([x_1[i:i+1] for i in range(batch_tensor.size(0)) if idx + i < len(dataset)])
        all_x2.extend([x_2[i:i+1] for i in range(batch_tensor.size(0)) if idx + i < len(dataset)])
        all_x3.extend([x_3[i:i+1] for i in range(batch_tensor.size(0)) if idx + i < len(dataset)])
        all_x4.extend([x_4[i:i+1] for i in range(batch_tensor.size(0)) if idx + i < len(dataset)])

# Save all images at the end
for i in tqdm(range(len(dataset))):
    tensor_to_pil(all_x1[i]).save(f"imagenet_64/samples_ibq/level_1/{i:05d}.png")
    tensor_to_pil(all_x2[i]).save(f"imagenet_64/samples_ibq/level_2/{i:05d}.png")
    tensor_to_pil(all_x3[i]).save(f"imagenet_64/samples_ibq/level_3/{i:05d}.png")
    tensor_to_pil(all_x4[i]).save(f"imagenet_64/samples_ibq/level_4/{i:05d}.png")
