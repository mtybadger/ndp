import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from modules.diffusionmodules.model import Encoder, Decoder
from modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from modules.vqvae.quantize import GumbelQuantize
from modules.vqvae.quantize import EMAVectorQuantizer
from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
import torch.optim.lr_scheduler as lr_scheduler

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig={
                     "double_z": False,
                     "z_channels": 256,
                     "resolution": 256,
                     "in_channels": 3,
                     "out_ch": 3,
                     "ch": 128,
                     "ch_mult": [1,1,2,2,4],
                     "num_res_blocks": 2,
                     "attn_resolutions": [16],
                     "dropout": 0.0
                 },
                 lossconfig={
                     "disc_conditional": False,
                     "disc_in_channels": 3,
                     "disc_start": 50000,
                     "disc_weight": 0.5,
                     "codebook_weight": 1.0
                 },
                 n_embed=8192,
                 embed_dim=8,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.automatic_optimization = False
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

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
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.9, 0.95))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.95))
        
        sched_ae = lr_scheduler.LinearLR(opt_ae, start_factor=1.0, end_factor=0.05, total_iters=50)
        sched_disc = lr_scheduler.LinearLR(opt_disc, start_factor=1.0, end_factor=0.05, total_iters=50)
        return [opt_ae, opt_disc], [sched_ae, sched_disc]
    
    def on_train_epoch_end(self, *args, **kwargs):
        lr_ae, lr_disc = self.lr_schedulers()
        lr_ae.step()
        lr_disc.step()

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


class VQMultiModel(pl.LightningModule):
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
                     "disc_start": 50000,
                     "disc_weight": 0.4,
                     "codebook_weight": 1.0
                 },
                 n_embed=8192,
                 embed_dim=8,
                 dropout_step=10000,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.automatic_optimization = False
        
        z_channels = 256
        
        self.encoder_1 = Encoder(**ddconfig, ch_mult=[1,1,2,2], in_channels=3, out_ch=3, z_channels=z_channels, resolution=256, attn_resolutions=[32])
        self.encoder_2 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=32, attn_resolutions=[16])
        self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[4])
        
        self.decoder_1 = Decoder(**ddconfig, ch_mult=[1,1,2,2], in_channels=712, out_ch=3, z_channels=z_channels * 3, resolution=256, attn_resolutions=[64])
        self.decoder_2 = Decoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels * 2, resolution=32, attn_resolutions=[16])
        self.decoder_3 = Decoder(**ddconfig, ch_mult=[1,1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[4])
        
        self.quantize_1 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_2 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_3 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        
        self.quant_conv_1 = torch.nn.Conv2d(z_channels * 3, embed_dim, 1)
        self.quant_conv_2 = torch.nn.Conv2d(z_channels * 2, embed_dim, 1)
        self.quant_conv_3 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        self.post_quant_conv_1 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_2 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_3 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
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

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        enc_1 = self.encoder_1(x)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        
        conv_3 = self.quant_conv_3(enc_3)
        quant_3, emb_loss_3, info_3 = self.quantize_3(conv_3)
        post_quant_3 = self.post_quant_conv_3(quant_3)
        
        dec_3 = self.decoder_3(post_quant_3)
        enc_2 = torch.cat([dec_3, enc_2], dim=1)
        
        conv_2 = self.quant_conv_2(enc_2)
        quant_2, emb_loss_2, info_2 = self.quantize_2(conv_2)
        post_quant_2 = self.post_quant_conv_2(quant_2)
        
        dec_2 = self.decoder_2(torch.cat([dec_3, post_quant_2], dim=1))
        enc_1 = torch.cat([dec_3.repeat(1, 1, 2, 2), dec_2, enc_1], dim=1)
        
        conv_1 = self.quant_conv_1(enc_1)
        quant_1, emb_loss_1, info_1 = self.quantize_1(conv_1)
        
        return quant_1, quant_2, quant_3, emb_loss_1 + emb_loss_2 + emb_loss_3, (info_1, info_2, info_3)

    def decode(self, quant1, quant2, quant3):
        
        quant2 = quant2.repeat(1, 1, 2, 2)
        quant3 = quant3.repeat(1, 1, 8, 8)
        
        if self.global_step < self.dropout_step:
            quant1 = self.post_quant_conv_1(quant1)
            quant2 = self.post_quant_conv_2(quant2)
            quant3 = self.post_quant_conv_3(quant3)
        else:
            # Randomly zero out 0-50% of quant3 and 0-25% of quant2
            mask3 = torch.rand_like(quant3) > torch.rand(1).item() * 0.5  # Random dropout between 0-50%
            mask2 = torch.rand_like(quant2) > torch.rand(1).item() * 0.25  # Random dropout between 0-25%
            quant3 = quant3 * mask3
            quant2 = quant2 * mask2
            
            quant1 = self.post_quant_conv_1(quant1)
            quant2 = self.post_quant_conv_2(quant2) 
            quant3 = self.post_quant_conv_3(quant3)
        return self.decoder_1(torch.cat([quant1, quant2, quant3], dim=1))
    
    def forward(self, input):
        quant1, quant2, quant3, diff, _ = self.encode(input)
        dec = self.decode(quant1, quant2, quant3)
        return dec, diff

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
        lr = self.learning_rate
        # List all parameters for the autoencoder optimizer:
        # 1. Encoder/decoder networks
        encoder_decoder_params = list(self.encoder_1.parameters()) + list(self.decoder_1.parameters()) + list(self.encoder_2.parameters()) + list(self.decoder_2.parameters()) + list(self.encoder_3.parameters()) + list(self.decoder_3.parameters())
        
        # 2. Quantization components
        quant_params = list(self.quantize_1.parameters()) + list(self.quantize_2.parameters()) + list(self.quantize_3.parameters())
        
        # 3. Pre/post quantization convolutions
        conv_params = list(self.quant_conv_1.parameters()) + list(self.quant_conv_2.parameters()) + list(self.quant_conv_3.parameters()) + list(self.post_quant_conv_1.parameters()) + list(self.post_quant_conv_2.parameters()) + list(self.post_quant_conv_3.parameters())
        
        # Combine all parameters
        opt_ae = torch.optim.Adam(
            encoder_decoder_params + quant_params + conv_params,
            lr=lr, betas=(0.9, 0.95)
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.95))
        
        sched_ae = lr_scheduler.LinearLR(opt_ae, start_factor=1.0, end_factor=0.05, total_iters=50)
        sched_disc = lr_scheduler.LinearLR(opt_disc, start_factor=1.0, end_factor=0.05, total_iters=50)
        
        return [opt_ae, opt_disc], [sched_ae, sched_disc]

    def get_last_layer(self):
        return self.decoder_1.conv_out.weight

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


# class GumbelVQ(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  temperature_scheduler_config,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  kl_weight=1e-8,
#                  remap=None,
#                  ):

#         z_channels = ddconfig["z_channels"]
#         super().__init__(ddconfig,
#                          lossconfig,
#                          n_embed,
#                          embed_dim,
#                          ckpt_path=None,
#                          ignore_keys=ignore_keys,
#                          image_key=image_key,
#                          colorize_nlabels=colorize_nlabels,
#                          monitor=monitor,
#                          )

#         self.loss.n_classes = n_embed
#         self.vocab_size = n_embed

#         self.quantize = GumbelQuantize(z_channels, embed_dim,
#                                        n_embed=n_embed,
#                                        kl_weight=kl_weight, temp_init=1.0,
#                                        remap=remap)

#         self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

#     def temperature_scheduling(self):
#         self.quantize.temperature = self.temperature_scheduler(self.global_step)

#     def encode_to_prequant(self, x):
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         return h

#     def decode_code(self, code_b):
#         raise NotImplementedError

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         self.temperature_scheduling()
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)

#         if optimizer_idx == 0:
#             # autoencode
#             aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")

#             self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return aeloss

#         if optimizer_idx == 1:
#             # discriminator
#             discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")
#             self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return discloss

#     def validation_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x, return_pred_indices=True)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
#                                         last_layer=self.get_last_layer(), split="val")

#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
#                                             last_layer=self.get_last_layer(), split="val")
#         rec_loss = log_dict_ae["val/rec_loss"]
#         self.log("val/rec_loss", rec_loss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log("val/aeloss", aeloss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict

#     def log_images(self, batch, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         # encode
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         quant, _, _ = self.quantize(h)
#         # decode
#         x_rec = self.decode(quant)
#         log["inputs"] = x
#         log["reconstructions"] = x_rec
#         return log


# class EMAVQ(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  remap=None,
#                  sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__(ddconfig,
#                          lossconfig,
#                          n_embed,
#                          embed_dim,
#                          ckpt_path=None,
#                          ignore_keys=ignore_keys,
#                          image_key=image_key,
#                          colorize_nlabels=colorize_nlabels,
#                          monitor=monitor,
#                          )
#         self.quantize = EMAVectorQuantizer(n_embed=n_embed,
#                                            embedding_dim=embed_dim,
#                                            beta=0.25,
#                                            remap=remap)
#     def configure_optimizers(self):
#         lr = self.learning_rate
#         #Remove self.quantize from parameter list since it is updated via EMA
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr, betas=(0.5, 0.9))
#         return [opt_ae, opt_disc], []                                           