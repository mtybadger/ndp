import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse

from data.custom import CustomTrain, CustomTest
from models.vqgan import VQMultiModel
from data.utils import custom_collate
from modules.losses.vqperceptual import VQLPIPSWithDiscriminatorInference

def process_dataset(dataset, batch_size, max_samples, prefix, rank=None):
    # Load previously saved data if it exists
    try:
        samples_processed = 0
        existing_data = []
        output_file = f'./imagenet/{prefix}.jsonl' if rank is None else f'./imagenet/{prefix}_rank{rank}.jsonl'
        with open(output_file, 'r') as f:
            for line in f:
                existing_data.append(json.loads(line))
        samples_processed = len(existing_data)
        print(f"Found {samples_processed} existing samples for {prefix}")
    except FileNotFoundError:
        existing_data = []
        samples_processed = 0
        print(f"No saved data found for {prefix}, calculating from start...")

    # -----------------------------
    # 1. Load your model
    # -----------------------------
    model = VQMultiModel.load_from_checkpoint("./imagenet/model/last.ckpt")
    model = model.half()
    model.to(f"cuda:{rank}")
    model.eval()
    # -----------------------------
    # 3. Prepare the LPIPS-like loss
    # -----------------------------
    loss_fn = VQLPIPSWithDiscriminatorInference().to(f"cuda:{rank}")

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=8, collate_fn=custom_collate)

    # Skip already processed batches
    batches_to_skip = samples_processed // batch_size

    # Load labels
    with open('./imagenet/labels_train.txt', 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    # If test set, override labels
    if prefix == "test":
        labels = [2] * len(labels)

    total_samples = 0

    # -----------------------------
    # 4. Loop over DataLoader
    # -----------------------------
    for batch_idx, batch in enumerate(tqdm(dataloader, total=max_samples // batch_size)):
        if batch_idx < batches_to_skip:
            continue
        if total_samples >= max_samples:
            break

        # Get inputs on GPU
        x = model.get_input(batch, model.image_key).to(f"cuda:{rank}").half()

        with torch.no_grad():
            _, _, _, _, diff, info = model.encode(x)

            # info is typically: ((_, _, info1), (_, _, info2), (_, _, info3), (_, _, info4))
            (_, _, info1), (_, _, info2), (_, _, info3), (_, _, info4) = info

            zero1 = model.quantize_1.get_codebook_entry(torch.zeros_like(info1), (x.shape[0], 16, 16, 8))
            zero2 = model.quantize_2.get_codebook_entry(torch.zeros_like(info2), (x.shape[0], 8, 8, 8))
            zero3 = model.quantize_3.get_codebook_entry(torch.zeros_like(info3), (x.shape[0], 4, 4, 8))

            quant1 = model.quantize_1.get_codebook_entry(info1, (x.shape[0], 16, 16, 8))
            quant2 = model.quantize_2.get_codebook_entry(info2, (x.shape[0], 8, 8, 8))
            quant3 = model.quantize_3.get_codebook_entry(info3, (x.shape[0], 4, 4, 8))
            quant4 = model.quantize_4.get_codebook_entry(info4, (x.shape[0], 2, 2, 8))

            # Decode with progressively more tokens
            y1 = model.decode(zero1, zero2, zero3, quant4)
            y2 = model.decode(zero1, zero2, quant3, quant4)
            y3 = model.decode(zero1, quant2, quant3, quant4)
            y4 = model.decode(quant1, quant2, quant3, quant4)

        # Concatenate images
        images = torch.cat([y1.unsqueeze(0), y2.unsqueeze(0),
                            y3.unsqueeze(0), y4.unsqueeze(0)], dim=0)

        # Build tokens
        # (Note: each info is offset by 1024, but do confirm your offsets)
        tokens = torch.cat([
            (info4.reshape(x.shape[0], -1) + 1024),
            (info3.reshape(x.shape[0], -1) + 2*1024),
            (info2.reshape(x.shape[0], -1) + 3*1024),
            (info1.reshape(x.shape[0], -1) + 4*1024),
        ], dim=1)

        # We'll store patch difficulties
        # Patch difficulties for 2x2, 4x4, 8x8
        patch_difficulties = [
            torch.zeros(x.shape[0], 2, 2).to(x.device),
            torch.zeros(x.shape[0], 4, 4).to(x.device),
            torch.zeros(x.shape[0], 8, 8).to(x.device),
        ]

        # -----------------------------
        # 5. Compute patch difficulties in parallel via loss_fn
        # -----------------------------
        # We can wrap each "loss" call in loss_fn(...) because it's also in DataParallel

        # 4th level: 2x2
        with torch.no_grad():
            base_losses = loss_fn(x, images[0])
            for i in range(2):
                for j in range(2):
                    detail = images[0].clone()
                    detail[:, :, 32*i:32*(i+1), 32*j:32*(j+1)] = x[:, :, 32*i:32*(i+1), 32*j:32*(j+1)]
                    detail_losses = loss_fn(x, detail)
                    patch_losses = base_losses - detail_losses
                    patch_difficulties[0][:, i, j] = patch_losses

            # 3rd level: 4x4
            base_losses = loss_fn(x, images[1])
            for i in range(4):
                for j in range(4):
                    detail = images[1].clone()
                    detail[:, :, 16*i:16*(i+1), 16*j:16*(j+1)] = x[:, :, 16*i:16*(i+1), 16*j:16*(j+1)]
                    detail_losses = loss_fn(x, detail)
                    patch_losses = base_losses - detail_losses
                    patch_difficulties[1][:, i, j] = patch_losses

            # 2nd level: 8x8
            base_losses = loss_fn(x, images[2])
            for i in range(8):
                for j in range(8):
                    detail = images[2].clone()
                    detail[:, :, 8*i:8*(i+1), 8*j:8*(j+1)] = x[:, :, 8*i:8*(i+1), 8*j:8*(j+1)]
                    detail_losses = loss_fn(x, detail)
                    patch_losses = base_losses - detail_losses
                    patch_difficulties[2][:, i, j] = patch_losses

            # Concatenate difficulties
            batch_difficulties = torch.cat([
                torch.zeros(x.shape[0], 1).to(x.device),                      # 1
                patch_difficulties[0].reshape(-1, 4),                         # 2x2 = 4
                patch_difficulties[1].reshape(-1, 16),                        # 4x4 = 16
                patch_difficulties[2].reshape(-1, 64),                        # 8x8 = 64
                torch.zeros(x.shape[0], 256).to(x.device)                     # 256
            ], dim=1)

        # -----------------------------
        # 6. Save each sample in the batch
        # -----------------------------
        output_file = f'./imagenet/{prefix}.jsonl' if rank is None else f'./imagenet/{prefix}_rank{rank}.jsonl'
        for i in range(x.shape[0]):
            current_sample_idx = samples_processed + i
            label = labels[current_sample_idx] if current_sample_idx < len(labels) else 0

            # Label as first token
            sample_tokens = [label] + tokens[i].cpu().numpy().tolist()

            sample = {
                "tokens": sample_tokens,
                "difficulties": batch_difficulties[i].cpu().numpy().tolist()
            }
            with open(output_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')

        total_samples += x.shape[0]
        samples_processed += x.shape[0]


def tokenize_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='Rank of process (0-7)')
    args = parser.parse_args()

    # if args.rank < 0 or args.rank > 7:
    #     raise ValueError("Rank must be between 0 and 7")

    # # Process training data
    # train_dataset = CustomTrain(
    #     training_images_list_file="./imagenet/train.txt",
    #     size=128
    # )
    # total_samples = len(train_dataset)
    # chunk_size = total_samples // 8
    # start_idx = args.rank * chunk_size
    # end_idx = start_idx + chunk_size if args.rank < 7 else total_samples
    
    # print(f"Processing rank {args.rank}, samples {start_idx} to {end_idx}")
    # process_dataset(train_dataset, batch_size=512, max_samples=end_idx-start_idx, prefix="train", rank=args.rank)

    # Process test data only on rank 0
    if args.rank == 0:
        test_dataset = CustomTest(
            test_images_list_file="./imagenet/test.txt",
            size=128
        )
        process_dataset(test_dataset, batch_size=512, max_samples=len(test_dataset), prefix="test", rank=args.rank)


if __name__ == "__main__":
    tokenize_data()
