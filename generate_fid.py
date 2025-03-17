import os
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from models.gpt import GPT, GPTConfig
import torchvision
import yaml
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Generate images with different CFG scales')
parser.add_argument('--cfg_scale', type=float, default=1.0,
                   help='Classifier-Free Guidance scale (default: 1.0)')
args = parser.parse_args()

# Load YAML file
with open('configs/gpt/imagenet_64_16_ndp.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_config_args = config['model']['init_args']['config']['init_args']
gpt_config = GPTConfig(**model_config_args)
tokenizer_config_args = config['model']['init_args']['tokenizer']['init_args']
tokenizer_class = config['model']['init_args']['tokenizer']['class_path']
module_path, class_name = tokenizer_class.rsplit('.', 1)
module = __import__(module_path, fromlist=[class_name])
TokenizerClass = getattr(module, class_name)
# Instantiate the tokenizer with the config arguments
tokenizer = TokenizerClass(**tokenizer_config_args)

# Create output directory if it doesn't exist
os.makedirs(f"./imagenet_64/samples_16_ndp_cfg{args.cfg_scale}/", exist_ok=True)
os.makedirs(f"./imagenet_64/samples_grid_16_ndp_cfg{args.cfg_scale}/", exist_ok=True)

# Add distributed setup
def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Initialize distributed setup if running in distributed mode
if "LOCAL_RANK" in os.environ:
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
else:
    local_rank = 0
    world_size = 1

# Move model to the correct device and wrap in DDP if distributed
model = GPT.load_from_checkpoint("imagenet_64/gpt_16_ndp/epoch=299-step=375600.ckpt", config=gpt_config, tokenizer=tokenizer)
model.eval()
model.half()
model.cuda(local_rank)
if world_size > 1:
    model = DDP(model, device_ids=[local_rank])

# Distribute classes across GPUs
classes_per_gpu = 1000 // world_size
start_class = local_rank * classes_per_gpu
end_class = start_class + classes_per_gpu if local_rank != world_size - 1 else 1000

# Generate images for all 1000 classes in batches of 512
num_images_per_class = 50
num_classes = 1000
batch_size = 128

# Calculate how many classes we can process in each batch
classes_per_batch = batch_size // num_images_per_class
# Handle the case where batch_size is smaller than num_images_per_class
classes_per_batch = max(1, classes_per_batch)

try:
    # Main generation loop
    for batch_start in tqdm(range(start_class, end_class, classes_per_batch), 
                           desc=f"Generating batches on GPU {local_rank}"):
        batch_end = min(batch_start + classes_per_batch, end_class)
        current_batch_classes = batch_end - batch_start
        current_batch_size = current_batch_classes * num_images_per_class
        
        # Create labels for all classes in this batch
        labels = torch.zeros((current_batch_size, 1), dtype=torch.long, device=model.device)
        positions = torch.zeros((current_batch_size, 1), dtype=torch.long, device=model.device)
        
        # Assign the correct class label to each group of images
        for i, class_idx in enumerate(range(batch_start, batch_end)):
            start_idx = i * num_images_per_class
            end_idx = start_idx + num_images_per_class
            labels[start_idx:end_idx, 0] = class_idx
        
        with torch.no_grad():
            if world_size > 1:
                # Access the underlying model through .module when using DDP
                content_tokens, positions = model.module.generate(labels, positions, max_new_tokens=340, 
                                                                temperature=1.0, cfg_scale=args.cfg_scale)
            else:
                content_tokens, positions = model.generate(labels, positions, max_new_tokens=340, 
                                                         temperature=1.0, cfg_scale=args.cfg_scale)

            if not (world_size > 1 and model.module.config.ndp or model.config.ndp):
                ind4 = content_tokens[:, 1:5]
                ind3 = content_tokens[:, 5:21]
                ind2 = content_tokens[:, 21:85]
                ind1 = content_tokens[:, 85:]
            else:
                positions = torch.roll(positions, shifts=1)
                positions[:, 0] = 0
                ind4 = torch.zeros((current_batch_size, 4), dtype=torch.int, device=model.device)
                ind3 = torch.zeros((current_batch_size, 16), dtype=torch.int, device=model.device)
                ind2 = torch.zeros((current_batch_size, 64), dtype=torch.int, device=model.device)
                ind1 = torch.zeros((current_batch_size, 256), dtype=torch.int, device=model.device)
                # For ind4 (positions 1-4)
                for b in range(current_batch_size):
                    # For ind4 (positions 1-4)
                    for pos_idx, target_pos in enumerate(range(1, 5)):
                        matches = positions[b] == target_pos
                        if matches.any():
                            # Get the last occurrence of this position
                            last_idx = torch.where(matches)[0][-1]
                            ind4[b, pos_idx] = content_tokens[b, last_idx]
                    
                    # For ind3 (positions 5-20)
                    for pos_idx, target_pos in enumerate(range(5, 21)):
                        matches = positions[b] == target_pos
                        if matches.any():
                            last_idx = torch.where(matches)[0][-1]
                            ind3[b, pos_idx] = content_tokens[b, last_idx]
                    
                    # For ind2 (positions 21-84)
                    for pos_idx, target_pos in enumerate(range(21, 85)):
                        matches = positions[b] == target_pos
                        if matches.any():
                            last_idx = torch.where(matches)[0][-1]
                            ind2[b, pos_idx] = content_tokens[b, last_idx]
                    
                    # For ind1 (positions 85-340)
                    for pos_idx, target_pos in enumerate(range(85, 341)):
                        matches = positions[b] == target_pos
                        if matches.any():
                            last_idx = torch.where(matches)[0][-1]
                            ind1[b, pos_idx] = content_tokens[b, last_idx]


            q1 = tokenizer.quantize.get_codebook_entry(ind1, shape=(current_batch_size, 16, 16, tokenizer.embed_dim))
            q2 = tokenizer.quantize.get_codebook_entry(ind2, shape=(current_batch_size, 8, 8, tokenizer.embed_dim))
            q3 = tokenizer.quantize.get_codebook_entry(ind3, shape=(current_batch_size, 4, 4, tokenizer.embed_dim))
            q4 = tokenizer.quantize.get_codebook_entry(ind4, shape=(current_batch_size, 2, 2, tokenizer.embed_dim))

            # Get the final images (fully decoded)
            images = tokenizer.decode(q1, q2, q3, q4).detach().cpu()
            images = torch.clamp(images, -1., 1.)
            
            # Save each individual image
            img_idx = 0
            for i, class_idx in enumerate(range(batch_start, batch_end)):
                class_images = images[i*num_images_per_class:(i+1)*num_images_per_class]
                
                # Save individual images
                for j in range(num_images_per_class):
                    img_save_path = os.path.join(f"./imagenet_64/samples_16_ndp_cfg{args.cfg_scale}/", 
                                                f"sample_{class_idx}_img_{j}.png")
                    save_image(class_images[j], img_save_path, normalize=True)
                
                # Create and save grid for this class
                grid = torchvision.utils.make_grid(class_images, nrow=10)  # 10 images per row
                save_path = os.path.join(f"./imagenet_64/samples_grid_16_ndp_cfg{args.cfg_scale}/", 
                                        f"sample_{class_idx}.png")
                save_image(grid, save_path, normalize=True)

finally:
    # Cleanup distributed process group
    if world_size > 1:
        dist.destroy_process_group()
