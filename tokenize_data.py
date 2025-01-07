from data.custom import CustomTrain, CustomTest
from models.vqgan import VQMultiModel
from data.utils import custom_collate
import torch
from torch.utils.data import DataLoader
import numpy as np
from modules.losses.vqperceptual import VQLPIPSWithDiscriminatorInference
from tqdm import tqdm
def tokenize_data():
    # Load previously saved difficulties if they exist and get number of processed samples
    try:
        existing_difficulties = np.loadtxt('./tinyimagenet/difficulties.txt')
        samples_processed = len(existing_difficulties)
        print(f"Found {samples_processed} existing samples")
    except:
        existing_difficulties = None
        samples_processed = 0
        print("No saved difficulties found, calculating from start...")
    
    model = VQMultiModel.load_from_checkpoint("./tinyimagenet/last-v3.ckpt")
    model.to("mps")
    model.eval()

    dataset = CustomTrain(
                training_images_list_file="./tinyimagenet/train.txt",
                size=64
            )

    batch_size = 100
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, collate_fn=custom_collate)
    loss = VQLPIPSWithDiscriminatorInference().to("mps")

    all_batch_difficulties = []
    if existing_difficulties is not None:
        all_batch_difficulties.append(existing_difficulties)
    
    # Skip already processed batches
    batches_to_skip = samples_processed // batch_size
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx < batches_to_skip:
            continue
            
        x = model.get_input(batch,model.image_key)

        with torch.no_grad():
            _, _, _, _, diff, info = model.encode(x.to("mps"))
            (_, _, info1), (_, _, info2), (_, _, info3), (_, _, info4) = info
            
            zero1 = model.quantize_1.get_codebook_entry(torch.zeros_like(info1), (batch_size, 16, 16, 8))
            zero2 = model.quantize_2.get_codebook_entry(torch.zeros_like(info2), (batch_size, 8, 8, 8))
            zero3 = model.quantize_3.get_codebook_entry(torch.zeros_like(info3), (batch_size, 4, 4, 8))
            
            quant1 = model.quantize_1.get_codebook_entry(info1, (batch_size, 16, 16, 8))
            quant2 = model.quantize_2.get_codebook_entry(info2, (batch_size, 8, 8, 8))
            quant3 = model.quantize_3.get_codebook_entry(info3, (batch_size, 4, 4, 8))
            quant4 = model.quantize_4.get_codebook_entry(info4, (batch_size, 2, 2, 8))
            
            # Generate reconstructions with progressively more tokens
            y1 = model.decode(zero1, zero2, zero3, quant4)
            y2 = model.decode(zero1, zero2, quant3, quant4)
            y3 = model.decode(zero1, quant2, quant3, quant4)
            y4 = model.decode(quant1, quant2, quant3, quant4)

        images = torch.cat([y1.unsqueeze(0), y2.unsqueeze(0), y3.unsqueeze(0), y4.unsqueeze(0)], dim=0)

        patch_difficulties = [torch.zeros(batch_size,2,2), torch.zeros(batch_size,4,4), torch.zeros(batch_size,8,8)]

        # fourth level 2x2
        patches = 0
        x = x.to("mps")
        base_losses = loss(x, images[0])
        for i in range(2):
            for j in range(2):
                detail = images[0].clone()
                detail[:, :, 32*i:32*(i+1), 32*j:32*(j+1)] = x[:, :, 32*i:32*(i+1), 32*j:32*(j+1)]
                    
                detail_losses = loss(x, detail)
                patch_losses = base_losses - detail_losses
                patch_difficulties[0][:,i,j] = patch_losses
                
                patches += 1

        # third level 4x4
        patches = 0
        base_losses = loss(x, images[1])
        for i in range(4):
            for j in range(4):
                detail = images[1].clone()
                detail[:, :, 16*i:16*(i+1), 16*j:16*(j+1)] = x[:, :, 16*i:16*(i+1), 16*j:16*(j+1)]
                    
                detail_losses = loss(x, detail)
                patch_losses = base_losses - detail_losses
                patch_difficulties[1][:,i,j] = patch_losses
                
                patches += 1
        # second level 8x8
        patches = 0
        base_losses = loss(x, images[2])
        for i in range(8):
            for j in range(8):
                detail = images[2].clone()
                detail[:, :, 8*i:8*(i+1), 8*j:8*(j+1)] = x[:, :, 8*i:8*(i+1), 8*j:8*(j+1)]
                    
                detail_losses = loss(x, detail)
                patch_losses = base_losses - detail_losses
                patch_difficulties[2][:,i,j] = patch_losses
                patches += 1
        
        batch_difficulties = torch.cat([
            patch_difficulties[0].reshape(-1, 4), # Level 1: 2x2
            patch_difficulties[1].reshape(-1, 16), # Level 2: 4x4 
            patch_difficulties[2].reshape(-1, 64)  # Level 3: 8x8
        ], dim=1)

        all_batch_difficulties.append(batch_difficulties.cpu().numpy())
        
        # Save progress after each batch
        current_difficulties = np.concatenate(all_batch_difficulties, axis=0)
        np.savetxt('./tinyimagenet/difficulties.txt', current_difficulties, fmt='%.6f')

    # Final difficulties are already saved
    all_difficulties = np.concatenate(all_batch_difficulties, axis=0)
    return all_difficulties
    
if __name__ == "__main__":
    tokenize_data()
