import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from modules.diffusionmodules.model import Encoder, Decoder
from modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from modules.vqvae.quantize import IndexPropagationQuantize
from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
import torch.optim.lr_scheduler as lr_scheduler

        
class IBQSharedModel(pl.LightningModule):
    def __init__(self,
                 ddconfig={
                     "double_z": False,
                     "ch": 128,
                     "num_res_blocks": 2,
                     "dropout": 0.0
                 },
                 lossconfig={
                     "disc_conditional": False,
                     "disc_in_channels": 3,
                     "disc_start": 200000,
                     "disc_weight": 0.5,
                     "codebook_weight": 0.5,
                     "entropy_weight": 0.5
                 },
                 n_embed=1024,
                 embed_dim=8,
                 dropout_step=40000,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 learning_rate=1.5e-4,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.automatic_optimization = False
        
        z_channels = 256
        
        self.encoder_1 = Encoder(**ddconfig, ch_mult=[1,2,4], in_channels=3, out_ch=3, z_channels=z_channels, resolution=64, attn_resolutions=[16])
        self.encoder_2 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[8])
        self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=8, attn_resolutions=[4])
        self.encoder_4 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=4, attn_resolutions=[2])
        
        self.decoder = Decoder(**ddconfig, ch_mult=[1,2,4], in_channels=z_channels, out_ch=3, z_channels=z_channels, resolution=64, attn_resolutions=[16])
        
        self.quantize = IndexPropagationQuantize(n_embed, embed_dim, beta=0.5, use_entropy_loss=True,
                                        remap=remap)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.5,
        #                                 remap=remap, sane_index_shape=sane_index_shape, l2_norm=True)
        
        # self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
        self.quant_conv_1 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_2 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_3 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_4 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        self.post_quant_conv_1 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_2 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_3 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_4 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
        self.dropout_step = dropout_step
        
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.learning_rate = learning_rate

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def encode(self, x):
        enc_1 = self.encoder_1(x)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)
        
        conv_4 = self.quant_conv_4(enc_4)
        quant_4, emb_loss_4, info_4 = self.quantize(conv_4)
        
        conv_3 = self.quant_conv_3(enc_3)
        quant_3, emb_loss_3, info_3 = self.quantize(conv_3)
        
        conv_2 = self.quant_conv_2(enc_2)
        quant_2, emb_loss_2, info_2 = self.quantize(conv_2)
        
        conv_1 = self.quant_conv_1(enc_1)
        quant_1, emb_loss_1, info_1 = self.quantize(conv_1)
        
        return quant_1, quant_2, quant_3, quant_4, emb_loss_1 + emb_loss_2 + emb_loss_3 + emb_loss_4, (info_1, info_2, info_3, info_4)

    def decode(self, quant1, quant2, quant3, quant4):
        # quant2 = quant2.repeat(1, 1, 2, 2)
        # quant3 = quant3.repeat(1, 1, 4, 4)
        # quant4 = quant4.repeat(1, 1, 8, 8)
        
        quant1 = self.post_quant_conv_1(quant1)
        quant2 = self.post_quant_conv_2(quant2)
        quant3 = self.post_quant_conv_3(quant3)
        quant4 = self.post_quant_conv_4(quant4)
        
        quant2 = F.interpolate(quant2, scale_factor=2, mode='nearest')
        quant3 = F.interpolate(quant3, scale_factor=4, mode='nearest')
        quant4 = F.interpolate(quant4, scale_factor=8, mode='nearest')
        
        out = quant1 + quant2 + quant3 + quant4
        out = self.decoder(out)
        return out
    
    def forward(self, input):
        quant1, quant2, quant3, quant4, diff, _ = self.encode(input)
        
        zero_1, zero_2, zero_3, zero_4 = self.get_zero_tokens(quant4, quant3, quant2, quant1, input.device)
                    
        y1 = self.decode(quant1, quant2, quant3, quant4)
        y2 = self.decode(zero_1, quant2, quant3, quant4)
        y3 = self.decode(zero_1, zero_2, quant3, quant4)
        y4 = self.decode(zero_1, zero_2, zero_3, quant4)
        return (y1, y2, y3, y4), diff
    
    def get_zero_tokens(self, B, C, device):
        
        zero_tokens = [
            self.quantize.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C))
        ]
        
        zero_1 = zero_tokens[0].repeat(1, 1, 16, 16)
        zero_2 = zero_tokens[1].repeat(1, 1, 8, 8)
        zero_3 = zero_tokens[2].repeat(1, 1, 4, 4)
        zero_4 = zero_tokens[3].repeat(1, 1, 2, 2)
        
        return zero_1, zero_2, zero_3, zero_4


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_ae, opt_disc = self.optimizers()
        
        # Get input
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        # First optimizer - autoencoder
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()

        # Second optimizer - discriminator
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        # Logging
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        
        # Log metrics only once
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log dictionaries without duplicate metrics
        for key, value in log_dict_ae.items():
            if key != "val/rec_loss":  # Skip already logged metrics
                self.log(key, value)
                
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = 1.5e-4
        # List all parameters for the autoencoder optimizer:
        # 1. Encoder/decoder networks
        encoder_decoder_params = list(self.encoder_1.parameters()) + list(self.encoder_2.parameters()) + list(self.encoder_3.parameters()) + list(self.encoder_4.parameters()) + list(self.decoder.parameters())
        
        # 2. Quantization components
        quant_params = list(self.quantize.parameters())
        
        # 3. Pre/post quantization convolutions
        conv_params = list(self.quant_conv_1.parameters()) + list(self.quant_conv_2.parameters()) + list(self.quant_conv_3.parameters()) + list(self.quant_conv_4.parameters()) + list(self.post_quant_conv_1.parameters()) + list(self.post_quant_conv_2.parameters()) + list(self.post_quant_conv_3.parameters()) + list(self.post_quant_conv_4.parameters())
        
        # Combine all parameters
        opt_ae = torch.optim.AdamW(
            encoder_decoder_params + quant_params + conv_params,
            lr=lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        
        sched_ae = lr_scheduler.LinearLR(opt_ae, start_factor=1.0, end_factor=0.05, total_iters=50)
        sched_disc = lr_scheduler.LinearLR(opt_disc, start_factor=1.0, end_factor=0.05, total_iters=50)
        
        return [opt_ae, opt_disc], [sched_ae, sched_disc]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
        
    def on_train_epoch_end(self, *args, **kwargs):
        lr_ae, lr_disc = self.lr_schedulers()
        lr_ae.step()
        lr_disc.step()

        
