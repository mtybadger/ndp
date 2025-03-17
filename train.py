# cli_main.py
import os
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from data.custom import CustomTrain, CustomTest
from torch.utils.data import DataLoader
from models.gpt import GPT, GPTConfig
from models.vqgan import IBQSharedModel
from datasets import load_dataset
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchvision import transforms
import torchvision

torch.set_float32_matmul_precision('high')


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, log_dir="/tmp"):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_dir = log_dir
        self.logger_log_images = {}
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        grid = torchvision.utils.make_grid(images, nrow=4)

        grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid*255).astype(np.uint8)
        filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
            split,
            global_step,
            current_epoch,
            batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "generate") and
                callable(pl_module.generate) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
                
            labels = torch.arange(0, self.max_images).reshape(self.max_images, 1)
            positions = torch.cat((torch.full((self.max_images, 1), 0, dtype=torch.int), torch.full((self.max_images, 1), 1, dtype=torch.int)), dim=1)
            
            z1, z2, z3, z4 = pl_module.tokenizer.get_zero_tokens(self.max_images, pl_module.tokenizer.embed_dim, pl_module.device)
            with torch.no_grad():
                content_tokens, positions = pl_module.generate(labels, positions, max_new_tokens=340, kv_caching=False)
                positions = positions[:, :-1]
                
            print('Got content tokens', content_tokens)
            print('Content tokens shape', content_tokens.shape)
            print('Got positions tokens', positions)
            print('Positions shape', positions.shape)
            
            if not pl_module.config.ndp:
                ind4 = content_tokens[:, 1:5]
                ind3 = content_tokens[:, 5:21]
                ind2 = content_tokens[:, 21:85]
                ind1 = content_tokens[:, 85:]
            else:
                # positions = torch.roll(positions, shifts=1)
                # positions[:, 0] = 0
                # print('Positions after roll', positions)
                ind4 = torch.zeros((self.max_images, 4), dtype=torch.int, device=pl_module.device)
                ind3 = torch.zeros((self.max_images, 16), dtype=torch.int, device=pl_module.device)
                ind2 = torch.zeros((self.max_images, 64), dtype=torch.int, device=pl_module.device)
                ind1 = torch.zeros((self.max_images, 256), dtype=torch.int, device=pl_module.device)
                # For each batch element, fill indices based on position tokens
                for b in range(self.max_images):
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
                
            q1 = pl_module.tokenizer.quantize.get_codebook_entry(ind1, shape=(self.max_images, 16, 16, pl_module.tokenizer.embed_dim))
            q2 = pl_module.tokenizer.quantize.get_codebook_entry(ind2, shape=(self.max_images, 8, 8, pl_module.tokenizer.embed_dim))
            q3 = pl_module.tokenizer.quantize.get_codebook_entry(ind3, shape=(self.max_images, 4, 4, pl_module.tokenizer.embed_dim))
            q4 = pl_module.tokenizer.quantize.get_codebook_entry(ind4, shape=(self.max_images, 2, 2, pl_module.tokenizer.embed_dim))
            
            images = [pl_module.tokenizer.decode(z1, z2, z3, q4), pl_module.tokenizer.decode(z1, z2, q3, q4), pl_module.tokenizer.decode(z1, q2, q3, q4), pl_module.tokenizer.decode(q1, q2, q3, q4)]
            images = torch.cat((images[0], images[1], images[2], images[3])).detach().cpu()
            print('Images shape', images.shape)

            if self.clamp:
                images = torch.clamp(images, -1., 1.)

            self.log_local(self.log_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class MyLightningCLI(LightningCLI):
    """Subclass LightningCLI so we can customize config parsing, callbacks, etc."""
    
    def after_instantiate_classes(self):
        """Runs after the DataModule/LightningModule/Trainer objects are created."""
        # Access the trainer, model, datamodule like so:
        trainer = self.trainer
        model = self.model
        datamodule = self.datamodule
        
        # configure learning rate
        bs, base_lr = datamodule.batch_size, 2e-7
        ngpu = trainer.num_devices
        accumulate_grad_batches = trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        
        # If you wanted to add or modify callbacks programmatically:
        # e.g. append your custom SetupCallback or ImageLogger:
        # trainer.callbacks.append(SetupCallback(...))
        # trainer.callbacks.append(ImageLogger(batch_frequency=10000, max_images=4, clamp=True))

        # If you want to set the learning rate dynamically:
        # e.g. model.learning_rate = ...
        pass
    
    from torch.utils.data import DataLoader

class ImageNet64DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Directly instantiate the datasets
        self.train_dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")["train"]
        self.val_dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")["test"]

    def prepare_data(self):
        # Nothing to prepare since datasets are instantiated in __init__
        pass

    def setup(self, stage=None):
        # Nothing to setup since datasets are instantiated in __init__
        pass
    
    def collate_fn(self, batch):
        # Initialize transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
        ])
        
        # Process each item in batch
        images = []
        labels = []
        for item in batch:
            # Apply transforms to PIL image
            img = transform(item['image'])
            images.append(img)
            labels.append(item['label'])
            
        # Stack into tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
            
        return {
            'images': images,
            'labels': labels
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return self.val_dataloader()



# class ImageNet64DataModuleTokenized(pl.LightningDataModule):
#     def __init__(self, tokenizer: IBQSharedModel, batch_size=8, num_workers=16, image_resolution=64, n_classes=1000, position_vocab_size=341):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.tokenizer = tokenizer
#         self.tokenizer.eval()
#         self.tokenizer.cpu()
#         # self.tokenizer.encode = torch.compile(self.tokenizer.encode)
#         # self.tokenizer.decode = torch.compile(self.tokenizer.decode)
#         for param in self.tokenizer.parameters():
#             param.requires_grad = False
#         self.image_resolution = image_resolution
#         self.n_classes = n_classes
#         self.position_vocab_size = position_vocab_size
#         # Directly instantiate the datasets
#         self.train_dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")["train"]
#         self.val_dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")["test"]

#     def prepare_data(self):
#         # Nothing to prepare since datasets are instantiated in __init__
#         pass

#     def setup(self, stage=None):
#         # Nothing to setup since datasets are instantiated in __init__
#         pass
    
#     def calculate_patchwise_loss(self, original, reconstructed, patch_size):
#         """Calculate MSE loss for each patch size"""
#         losses = {}
#         batch, channels, height, width = original.shape
        
#         # Calculate number of patches
#         h_patches = height // patch_size
#         w_patches = width // patch_size
        
#         # Reshape tensors to extract all patches at once
#         # [B, C, H, W] -> [B, C, h_patches, patch_size, w_patches, patch_size]
#         orig_patches = original.view(batch, channels, h_patches, patch_size, w_patches, patch_size)
#         recon_patches = reconstructed.view(batch, channels, h_patches, patch_size, w_patches, patch_size)
        
#         # Permute and reshape to get all patches: [B, h_patches*w_patches, C, patch_size, patch_size]
#         orig_patches = orig_patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, h_patches*w_patches, -1)
#         recon_patches = recon_patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, h_patches*w_patches, -1)
        
#         # Calculate MSE for all patches simultaneously
#         patch_losses = torch.sqrt(((orig_patches - recon_patches) ** 2).sum(dim=2))
        
#         return patch_losses
    
#     def collate_fn(self, batch):
#         # Initialize transforms
#         transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
#             transforms.ToTensor(),
#         ])
        
#         # Process each item in batch
#         images = []
#         labels = []
#         for item in batch:
#             # Apply transforms to PIL image
#             img = transform(item['image'])
#             images.append(img)
#             labels.append(item['label'])
            
#         # Stack into tensors
#         images = torch.stack(images)
#         labels = torch.tensor(labels)
            
#         with torch.inference_mode():
#                 q1, q2, q3, q4, _, ((_, _, ind1), (_, _, ind2), (_, _, ind3), (_, _, ind4)) = self.tokenizer.encode(images)
        
#         ind1 = torch.reshape(ind1, (images.size(0), -1))
#         ind2 = torch.reshape(ind2, (images.size(0), -1))
#         ind3 = torch.reshape(ind3, (images.size(0), -1))
#         ind4 = torch.reshape(ind4, (images.size(0), -1))
#         class_token = labels.unsqueeze(1)
#         if torch.rand(1) < 0.1:
#             class_token = torch.full((labels.size(0), 1), self.n_classes, device=labels.device)
#         with torch.inference_mode():
#                 z1, z2, z3, z4 = self.tokenizer.get_zero_tokens(1, self.tokenizer.embed_dim, labels.device)
#                 losses_4 = self.calculate_patchwise_loss(images, self.tokenizer.decode(z1, z2, z3, q4), self.image_resolution // 2)
#                 losses_3 = self.calculate_patchwise_loss(images, self.tokenizer.decode(z1, z2, q3, q4), self.image_resolution // 4)
#                 losses_2 = self.calculate_patchwise_loss(images, self.tokenizer.decode(z1, q2, q3, q4), self.image_resolution // 8)
#                 losses_1 = self.calculate_patchwise_loss(images, self.tokenizer.decode(q1, q2, q3, q4), self.image_resolution // 16)   
#                 all_losses = torch.cat((torch.full((losses_4.size(0), 1), float('inf'), device=labels.device, dtype=losses_4.dtype), losses_4, losses_3, losses_2, losses_1), dim=1)
                
#                 sorted_indices = torch.argsort(all_losses, dim=1, descending=True)
#                 content_tokens = torch.cat((class_token, ind4, ind3, ind2, ind1), dim=1)
#                 position_tokens = torch.arange(content_tokens.size(1)).unsqueeze(0).expand(content_tokens.size(0), -1).to(labels.device)
                
#                 content_tokens = torch.gather(content_tokens, dim=1, index=sorted_indices)
#                 position_tokens = torch.gather(position_tokens, dim=1, index=sorted_indices)
#                 position_targets = torch.cat((position_tokens[:, 2:], torch.full((position_tokens.size(0), 2), -100, device=position_tokens.device, dtype=torch.long)), dim=1)
#                 position_tokens = torch.cat((position_tokens[:, 1:], torch.full((position_tokens.size(0), 1), self.position_vocab_size - 1, device=position_tokens.device, dtype=torch.long)), dim=1)
#                 content_targets = torch.cat((content_tokens[:, 1:], torch.full((content_tokens.size(0), 1), -100, device=content_tokens.device, dtype=torch.long)), dim=1)
                
#         return (content_tokens, position_tokens, content_targets, position_targets)

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             collate_fn=self.collate_fn,
#             persistent_workers=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             collate_fn=self.collate_fn,
#             persistent_workers=True
#         )

#     def test_dataloader(self):
#         return self.val_dataloader()




def main():
    # By default, the CLI subcommands are: fit, validate, test, predict
    # If you call `MyLightningCLI(..., run=True)`, it will automatically parse
    # and then run the subcommand. For example, `python cli_main.py fit --model.x=123`.
    MyLightningCLI(
        seed_everything_default=23,             # default seed
        save_config_callback=None,              # turn off saving the config if you want
        trainer_defaults={
            "callbacks": [
                LearningRateMonitor(logging_interval="step"),
            ],
            "default_root_dir": os.getcwd(),
            "max_epochs": 40,
            "precision": "bf16-mixed",
            "log_every_n_steps": 10,
        },
        run=True,  # parse args + run subcommand (fit, test, etc.)
    )


if __name__ == "__main__":
    main()
