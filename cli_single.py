# cli_main.py
import os
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
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
import torchvision

# torch.set_float32_matmul_precision('medium')

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {}
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                if isinstance(images[k], (list, tuple)):
                    # Concatenate all images in the tuple vertically
                    N = min(images[k][0].shape[0], self.max_images)
                    processed_images = []
                    for img in images[k]:
                        img = img[:N]
                        if isinstance(img, torch.Tensor):
                            img = img.detach().cpu()
                            if self.clamp:
                                img = torch.clamp(img, -1., 1.)
                        processed_images.append(img)
                    images[k] = torch.cat(processed_images, dim=2)  # Concatenate along height dimension
                else:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local('/home/ubuntu/cifar-testing/ndp/tinyimagenet/single_logs_2/', split, images,
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
        
        self.trainer.logger = WandbLogger(project="vqgan_single")
        
        # configure learning rate
        bs, base_lr = datamodule.batch_size, 2e-6
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
        trainer.callbacks.append(ImageLogger(batch_frequency=10000, max_images=4, clamp=True))

        # If you want to set the learning rate dynamically:
        # e.g. model.learning_rate = ...
        pass
    
    from torch.utils.data import DataLoader

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128,
                 wrap=False, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wrap = wrap

        # Directly instantiate the datasets
        self.train_dataset = CustomTrain(
            training_images_list_file="/home/ubuntu/cifar-testing/ndp/cifar10/train.txt",
            size=32
        )
        
        self.val_dataset = CustomTest(
            test_images_list_file="/home/ubuntu/cifar-testing/ndp/cifar10/test.txt", 
            size=32
        )

        if self.wrap:
            self.train_dataset = WrappedDataset(self.train_dataset)
            self.val_dataset = WrappedDataset(self.val_dataset)

    def prepare_data(self):
        # Nothing to prepare since datasets are instantiated in __init__
        pass

    def setup(self, stage=None):
        # Nothing to setup since datasets are instantiated in __init__
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )

    def test_dataloader(self):
        return self.val_dataloader()


class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128,
                 wrap=False, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wrap = wrap

        # Directly instantiate the datasets
        self.train_dataset = CustomTrain(
            training_images_list_file="/home/ubuntu/cifar-testing/ndp/tinyimagenet/train.txt",
            size=64
        )
        
        self.val_dataset = CustomTest(
            test_images_list_file="/home/ubuntu/cifar-testing/ndp/tinyimagenet/test.txt", 
            size=64
        )

        if self.wrap:
            self.train_dataset = WrappedDataset(self.train_dataset)
            self.val_dataset = WrappedDataset(self.val_dataset)

    def prepare_data(self):
        # Nothing to prepare since datasets are instantiated in __init__
        pass

    def setup(self, stage=None):
        # Nothing to setup since datasets are instantiated in __init__
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate
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
