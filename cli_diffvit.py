# cli_main.py
import os
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from datasets import load_dataset
import torch
from main import (
    # SetupCallback,
    WrappedDataset,
)
from data.utils import custom_collate
from data.custom import CustomTrain, CustomTest
from torch.utils.data import DataLoader
from models.vqgan import VQMultiModel
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from models.vit import DifficultyViT
import torchvision


class MyLightningCLI(LightningCLI):
    """Subclass LightningCLI so we can customize config parsing, callbacks, etc."""
    
    def add_arguments_to_parser(self, parser):
        """Use this hook to add or modify arguments at the CLI level."""
        parser.add_argument("--debug", action="store_true", default=False,
                           help="Enable debug mode with fewer steps, offline logger, etc.")
        parser.add_argument("--seed", type=int, default=23, help="Random seed")

        # You can also modify existing subparsers (model, data, trainer) here
        # by calling parser.set_defaults(...). 
        # See: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_usage.html

    def before_instantiate_classes(self):
        """No need to handle nested configs since we flattened the parameters"""
        pass

    def after_instantiate_classes(self):
        """Runs after the DataModule/LightningModule/Trainer objects are created."""
        # Access the trainer, model, datamodule like so:
        trainer = self.trainer
        model = self.model
        datamodule = self.datamodule
        
        self.trainer.logger = WandbLogger(project="diffvit")
        
        # configure learning rate
        bs, base_lr = datamodule.batch_size, 2e-6
        ngpu = trainer.num_devices
        accumulate_grad_batches = trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        pass
    
    from torch.utils.data import DataLoader

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128,
                 wrap=False, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset = load_dataset("json", data_files={
            "train": "./tinyimagenet/train.jsonl",
            "test": "./tinyimagenet/test.jsonl"
        }).with_format("torch")
        print(self.dataset)
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return self.val_dataloader()



def main():
    # By default, the CLI subcommands are: fit, validate, test, predict
    # If you call `MyLightningCLI(..., run=True)`, it will automatically parse
    # and then run the subcommand. For example, `python cli_main.py fit --model.x=123`.
    MyLightningCLI(
        run=True,  # parse args + run subcommand (fit, test, etc.)
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    main()
