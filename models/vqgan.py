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
                     "disc_start": 200000,
                     "disc_weight": 0.5,
                     "codebook_weight": 0.5
                 },
                 n_embed=512,
                 embed_dim=8,
                 dropout_step=40000,
                 token_drop=[32, 8, 2, 0],
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
        
        self.encoder_1 = Encoder(**ddconfig, ch_mult=[1,2,2], in_channels=3, out_ch=3, z_channels=z_channels, resolution=64, attn_resolutions=[16])
        self.encoder_2 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[8])
        self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=8, attn_resolutions=[4])
        self.encoder_4 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=4, attn_resolutions=[2])
        
        self.decoder = Decoder(**ddconfig, ch_mult=[1,2,2], in_channels=z_channels * 4, out_ch=3, z_channels=z_channels * 4, resolution=64, attn_resolutions=[16])
        
        self.quantize_1 = VectorQuantizer(n_embed, embed_dim, beta=0.5,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_norm=True)
        self.quantize_2 = VectorQuantizer(n_embed, embed_dim, beta=0.5,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_norm=True)
        self.quantize_3 = VectorQuantizer(n_embed, embed_dim, beta=0.5,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_norm=True)
        self.quantize_4 = VectorQuantizer(n_embed, embed_dim, beta=0.5,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_norm=True)
        
        self.quant_conv_1 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_2 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_3 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_4 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        self.post_quant_conv_1 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_2 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_3 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_4 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
        self.dropout_step = dropout_step
        
        self.token_drop = token_drop
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
        enc_4 = self.encoder_4(enc_3)
        
        conv_4 = self.quant_conv_4(enc_4)
        quant_4, emb_loss_4, info_4 = self.quantize_4(conv_4)
        
        conv_3 = self.quant_conv_3(enc_3)
        quant_3, emb_loss_3, info_3 = self.quantize_3(conv_3)
        
        conv_2 = self.quant_conv_2(enc_2)
        quant_2, emb_loss_2, info_2 = self.quantize_2(conv_2)
        
        conv_1 = self.quant_conv_1(enc_1)
        quant_1, emb_loss_1, info_1 = self.quantize_1(conv_1)
        
        return quant_1, quant_2, quant_3, quant_4, emb_loss_1 + emb_loss_2 + emb_loss_3 + emb_loss_4, (info_1, info_2, info_3, info_4)

    def decode(self, quant1, quant2, quant3, quant4):
        quant2 = quant2.repeat(1, 1, 2, 2)
        quant3 = quant3.repeat(1, 1, 4, 4)
        quant4 = quant4.repeat(1, 1, 8, 8)
        
        quant1 = self.post_quant_conv_1(quant1)
        quant2 = self.post_quant_conv_2(quant2)
        quant3 = self.post_quant_conv_3(quant3)
        quant4 = self.post_quant_conv_4(quant4)
        
        out = torch.cat([quant1, quant2, quant3, quant4], dim=1)
        out = self.decoder(out)
        return out
    
    def forward(self, input):
        quant1, quant2, quant3, quant4, diff, _ = self.encode(input)
        
        (quant1, quant2, quant3, quant4), (zero_1, zero_2, zero_3, zero_4) = self.cascade_drop(quant4, quant3, quant2, quant1, input.device, self.token_drop)
                    
        y1 = self.decode(quant1, quant2, quant3, quant4)
        y2 = self.decode(zero_1, quant2, quant3, quant4)
        y3 = self.decode(zero_1, zero_2, quant3, quant4)
        y4 = self.decode(zero_1, zero_2, zero_3, quant4)
        return (y1, y2, y3, y4), diff
    
    def cascade_drop(self, quant4, quant3, quant2, quant1, device, token_drop):
        
        
        B, C, H1, W1 = quant1.shape  # 16x16
        _, _, H2, W2 = quant2.shape  # 8x8
        _, _, H3, W3 = quant3.shape  # 4x4
        _, _, H4, W4 = quant4.shape  # 2x2
        
        zero_tokens = [
            self.quantize_1.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_2.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_3.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_4.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C))
        ]
        
        # --------------------------
        # 2) Dropout in quant3 (4x4)
        # --------------------------
        num_drop_3 = torch.randint(0, token_drop[2]+1, (1,)).item()
        if num_drop_3 > 0:
            indices_3 = torch.randperm(H3*W3)[:num_drop_3].to(device)
            for idx in indices_3:
                c_h = idx // W3
                c_w = idx % W3
                quant3[:, :, c_h, c_w] = zero_tokens[2][:, :, 0, 0]
                
                # zero out region in quant2 -> scale factor is 2
                block_size_2 = 2
                r_start_2 = block_size_2 * c_h
                c_start_2 = block_size_2 * c_w
                quant2[:, :, r_start_2:r_start_2+block_size_2, c_start_2:c_start_2+block_size_2] = zero_tokens[1]
                
                # zero out region in quant1 -> scale factor is 4
                block_size_1 = 4
                r_start_1 = block_size_1 * c_h
                c_start_1 = block_size_1 * c_w
                quant1[:, :, r_start_1:r_start_1+block_size_1, c_start_1:c_start_1+block_size_1] = zero_tokens[0]
        
        # --------------------------
        # 3) Dropout in quant2 (8x8)
        # --------------------------
        num_drop_2 = torch.randint(0, token_drop[1]+1, (1,)).item()
        if num_drop_2 > 0:
            indices_2 = torch.randperm(H2*W2)[:num_drop_2].to(device)
            for idx in indices_2:
                c_h = idx // W2
                c_w = idx % W2
                quant2[:, :, c_h, c_w] = zero_tokens[1][:, :, 0, 0]
                
                # zero out region in quant1 -> scale factor is 2
                block_size_1 = 2
                r_start_1 = block_size_1 * c_h
                c_start_1 = block_size_1 * c_w
                quant1[:, :, r_start_1:r_start_1+block_size_1, c_start_1:c_start_1+block_size_1] = zero_tokens[0]
                
        # --------------------------
        # 4) Dropout in quant1 (16x16)
        # --------------------------
        num_drop_1 = torch.randint(0, token_drop[0]+1, (1,)).item()
        if num_drop_1 > 0:
            indices_1 = torch.randperm(H1*W1)[:num_drop_1].to(device)
            for idx in indices_1:
                c_h = idx // W1
                c_w = idx % W1
                quant1[:, :, c_h, c_w] = zero_tokens[0][:, :, 0, 0]
                
        zero_1 = zero_tokens[0].repeat(1, 1, 16, 16)
        zero_2 = zero_tokens[1].repeat(1, 1, 8, 8)
        zero_3 = zero_tokens[2].repeat(1, 1, 4, 4)
        zero_4 = zero_tokens[3].repeat(1, 1, 2, 2)
        
        return (quant1, quant2, quant3, quant4), (zero_1, zero_2, zero_3, zero_4)


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
        encoder_decoder_params = list(self.encoder_1.parameters()) + list(self.encoder_2.parameters()) + list(self.encoder_3.parameters()) + list(self.encoder_4.parameters()) + list(self.decoder.parameters())
        
        # 2. Quantization components
        quant_params = list(self.quantize_1.parameters()) + list(self.quantize_2.parameters()) + list(self.quantize_3.parameters()) + list(self.quantize_4.parameters())
        
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


    # def configure_optimizers(self):
    #     lr = self.learning_rate
        
    #     # -- STEP 1/2: Separate Conv2d params & others for the autoencoder --
    #     conv_params_ae = []
    #     other_params_ae = []
    #     # "encoder_decoder_params + quant_params + conv_params" is effectively all AE params,
    #     # but let's iterate modules for clarity and avoid shape-based filtering.
    #     for module in [
    #         self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4,
    #         self.decoder,
    #         self.quantize_1, self.quantize_2, self.quantize_3, self.quantize_4,
    #         self.quant_conv_1, self.quant_conv_2, self.quant_conv_3, self.quant_conv_4,
    #         self.post_quant_conv_1, self.post_quant_conv_2, self.post_quant_conv_3, self.post_quant_conv_4
    #     ]:
    #         for submodule in module.modules():
    #             if isinstance(submodule, torch.nn.Conv2d):
    #                 conv_params_ae += list(submodule.parameters())
    #             else:
    #                 other_params_ae += list(submodule.parameters())

    #     # -- STEP 1/2 (continued): Create the Muon optimizer for all conv2d, AdamW-like for everything else --
    #     opt_ae = Muon(
    #         muon_params=conv_params_ae,             # the conv weights
    #         lr=0.02,                              # you can set your Muon LR
    #         momentum=0.95,
    #         adamw_params=other_params_ae,         # "everything else"
    #         adamw_lr=lr,                          # AdamW LR
    #         adamw_betas=(0.90, 0.95),
    #         adamw_wd=0.01
    #     )

    #     # -- STEP 2/2: Repeat the same idea for the discriminator --
    #     conv_params_disc = []
    #     other_params_disc = []
    #     for submodule in self.loss.discriminator.modules():
    #         if isinstance(submodule, torch.nn.Conv2d):
    #             conv_params_disc += list(submodule.parameters())
    #         else:
    #             other_params_disc += list(submodule.parameters())

    #     opt_disc = Muon(
    #         muon_params=conv_params_disc,
    #         lr=0.02,
    #         momentum=0.95,
    #         adamw_params=other_params_disc,
    #         adamw_lr=lr,
    #         adamw_betas=(0.90, 0.95),
    #         adamw_wd=0.01
    #     )

    #     # Return both optimizers
    #     return [opt_ae, opt_disc], []

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
        
class VQHierarchicalMultiModel(pl.LightningModule):
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
                     "codebook_weight": 0.5
                 },
                 n_embed=4096,
                 embed_dim=8,
                 dropout_step=40000,
                 ckpt_path=None,
                 token_drop=[32, 8, 2, 0],
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
        
        self.encoder_1 = Encoder(**ddconfig, ch_mult=[1,2,2], in_channels=3, out_ch=3, z_channels=z_channels, resolution=64, attn_resolutions=[16])
        self.encoder_2 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[8])
        self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=8, attn_resolutions=[4])
        self.encoder_4 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=4, attn_resolutions=[2])
        
        self.decoder_4 = Decoder(**ddconfig, ch_mult=[1,2], in_channels=z_channels, out_ch=z_channels, z_channels=z_channels, resolution=4, attn_resolutions=[2])
        self.decoder_3 = Decoder(**ddconfig, ch_mult=[1,2], in_channels=z_channels * 2, out_ch=z_channels, z_channels=z_channels * 2, resolution=8, attn_resolutions=[4])
        self.decoder_2 = Decoder(**ddconfig, ch_mult=[1,2], in_channels=z_channels * 3, out_ch=z_channels, z_channels=z_channels * 3, resolution=16, attn_resolutions=[8])
        self.decoder_1 = Decoder(**ddconfig, ch_mult=[1,2,2], in_channels=z_channels * 4, out_ch=3, z_channels=z_channels * 4, resolution=64, attn_resolutions=[16])
        
        self.quantize_1 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_2 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_3 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_4 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        
        self.quant_conv_1 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_2 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_3 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_4 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        self.post_quant_conv_1 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_2 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_3 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.post_quant_conv_4 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
        self.dropout_step = dropout_step
        self.token_drop = token_drop
        
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
        enc_4 = self.encoder_4(enc_3)
        
        conv_4 = self.quant_conv_4(enc_4)
        quant_4, emb_loss_4, info_4 = self.quantize_4(conv_4)
        
        conv_3 = self.quant_conv_3(enc_3)
        quant_3, emb_loss_3, info_3 = self.quantize_3(conv_3)
        
        conv_2 = self.quant_conv_2(enc_2)
        quant_2, emb_loss_2, info_2 = self.quantize_2(conv_2)
        
        conv_1 = self.quant_conv_1(enc_1)
        quant_1, emb_loss_1, info_1 = self.quantize_1(conv_1)
        
        return quant_1, quant_2, quant_3, quant_4, emb_loss_1 + emb_loss_2 + emb_loss_3 + emb_loss_4, (info_1, info_2, info_3, info_4)

    def decode(self, quant1, quant2, quant3, quant4):
        quant1 = self.post_quant_conv_1(quant1)
        quant2 = self.post_quant_conv_2(quant2)
        quant3 = self.post_quant_conv_3(quant3)
        quant4 = self.post_quant_conv_4(quant4)
        
        dec_4 = self.decoder_4(quant4)
        dec_3 = self.decoder_3(torch.cat([quant3, dec_4], dim=1))
        dec_2 = self.decoder_2(torch.cat([quant2, dec_3, dec_4.repeat(1,1,2,2)], dim=1))
        dec_1 = self.decoder_1(torch.cat([quant1, dec_2, dec_3.repeat(1,1,2,2), dec_4.repeat(1,1,4,4)], dim=1))
        
        return dec_1
    
    def forward(self, input):
        quant1, quant2, quant3, quant4, diff, _ = self.encode(input)
        
        (quant1, quant2, quant3, quant4), (zero_1, zero_2, zero_3, zero_4) = self.cascade_drop(quant4, quant3, quant2, quant1, input.device, self.token_drop)
                    
        y1 = self.decode(quant1, quant2, quant3, quant4)
        y2 = self.decode(zero_1, quant2, quant3, quant4)
        y3 = self.decode(zero_1, zero_2, quant3, quant4)
        y4 = self.decode(zero_1, zero_2, zero_3, quant4)
        return (y1, y2, y3, y4), diff
    
    def cascade_drop(self, quant4, quant3, quant2, quant1, device, token_drop):
        
        
        B, C, H1, W1 = quant1.shape  # 16x16
        _, _, H2, W2 = quant2.shape  # 8x8
        _, _, H3, W3 = quant3.shape  # 4x4
        _, _, H4, W4 = quant4.shape  # 2x2
        
        zero_tokens = [
            self.quantize_1.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_2.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_3.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C)),
            self.quantize_4.get_codebook_entry(torch.zeros(B, dtype=torch.long, device=device), shape=(B, 1, 1, C))
        ]
        
        # --------------------------
        # 2) Dropout in quant3 (4x4)
        # --------------------------
        num_drop_3 = torch.randint(0, token_drop[2]+1, (1,)).item()
        if num_drop_3 > 0:
            indices_3 = torch.randperm(H3*W3)[:num_drop_3].to(device)
            for idx in indices_3:
                c_h = idx // W3
                c_w = idx % W3
                quant3[:, :, c_h, c_w] = zero_tokens[2][:, :, 0, 0]
                
                # zero out region in quant2 -> scale factor is 2
                block_size_2 = 2
                r_start_2 = block_size_2 * c_h
                c_start_2 = block_size_2 * c_w
                quant2[:, :, r_start_2:r_start_2+block_size_2, c_start_2:c_start_2+block_size_2] = zero_tokens[1]
                
                # zero out region in quant1 -> scale factor is 4
                block_size_1 = 4
                r_start_1 = block_size_1 * c_h
                c_start_1 = block_size_1 * c_w
                quant1[:, :, r_start_1:r_start_1+block_size_1, c_start_1:c_start_1+block_size_1] = zero_tokens[0]
        
        # --------------------------
        # 3) Dropout in quant2 (8x8)
        # --------------------------
        num_drop_2 = torch.randint(0, token_drop[1]+1, (1,)).item()
        if num_drop_2 > 0:
            indices_2 = torch.randperm(H2*W2)[:num_drop_2].to(device)
            for idx in indices_2:
                c_h = idx // W2
                c_w = idx % W2
                quant2[:, :, c_h, c_w] = zero_tokens[1][:, :, 0, 0]
                
                # zero out region in quant1 -> scale factor is 2
                block_size_1 = 2
                r_start_1 = block_size_1 * c_h
                c_start_1 = block_size_1 * c_w
                quant1[:, :, r_start_1:r_start_1+block_size_1, c_start_1:c_start_1+block_size_1] = zero_tokens[0]
                
        # --------------------------
        # 4) Dropout in quant1 (16x16)
        # --------------------------
        num_drop_1 = torch.randint(0, token_drop[0]+1, (1,)).item()
        if num_drop_1 > 0:
            indices_1 = torch.randperm(H1*W1)[:num_drop_1].to(device)
            for idx in indices_1:
                c_h = idx // W1
                c_w = idx % W1
                quant1[:, :, c_h, c_w] = zero_tokens[0][:, :, 0, 0]
                
        zero_1 = zero_tokens[0].repeat(1, 1, 16, 16)
        zero_2 = zero_tokens[1].repeat(1, 1, 8, 8)
        zero_3 = zero_tokens[2].repeat(1, 1, 4, 4)
        zero_4 = zero_tokens[3].repeat(1, 1, 2, 2)
        
        return (quant1, quant2, quant3, quant4), (zero_1, zero_2, zero_3, zero_4)


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
        encoder_decoder_params = list(self.encoder_1.parameters()) + list(self.encoder_2.parameters()) + list(self.encoder_3.parameters()) + list(self.encoder_4.parameters()) + list(self.decoder_1.parameters()) + list(self.decoder_2.parameters()) + list(self.decoder_3.parameters()) + list(self.decoder_4.parameters())
        
        # 2. Quantization components
        quant_params = list(self.quantize_1.parameters()) + list(self.quantize_2.parameters()) + list(self.quantize_3.parameters()) + list(self.quantize_4.parameters())
        
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


    # def configure_optimizers(self):
    #     lr = self.learning_rate
        
    #     # -- STEP 1/2: Separate Conv2d params & others for the autoencoder --
    #     conv_params_ae = []
    #     other_params_ae = []
    #     # "encoder_decoder_params + quant_params + conv_params" is effectively all AE params,
    #     # but let's iterate modules for clarity and avoid shape-based filtering.
    #     for module in [
    #         self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4,
    #         self.decoder,
    #         self.quantize_1, self.quantize_2, self.quantize_3, self.quantize_4,
    #         self.quant_conv_1, self.quant_conv_2, self.quant_conv_3, self.quant_conv_4,
    #         self.post_quant_conv_1, self.post_quant_conv_2, self.post_quant_conv_3, self.post_quant_conv_4
    #     ]:
    #         for submodule in module.modules():
    #             if isinstance(submodule, torch.nn.Conv2d):
    #                 conv_params_ae += list(submodule.parameters())
    #             else:
    #                 other_params_ae += list(submodule.parameters())

    #     # -- STEP 1/2 (continued): Create the Muon optimizer for all conv2d, AdamW-like for everything else --
    #     opt_ae = Muon(
    #         muon_params=conv_params_ae,             # the conv weights
    #         lr=0.02,                              # you can set your Muon LR
    #         momentum=0.95,
    #         adamw_params=other_params_ae,         # "everything else"
    #         adamw_lr=lr,                          # AdamW LR
    #         adamw_betas=(0.90, 0.95),
    #         adamw_wd=0.01
    #     )

    #     # -- STEP 2/2: Repeat the same idea for the discriminator --
    #     conv_params_disc = []
    #     other_params_disc = []
    #     for submodule in self.loss.discriminator.modules():
    #         if isinstance(submodule, torch.nn.Conv2d):
    #             conv_params_disc += list(submodule.parameters())
    #         else:
    #             other_params_disc += list(submodule.parameters())

    #     opt_disc = Muon(
    #         muon_params=conv_params_disc,
    #         lr=0.02,
    #         momentum=0.95,
    #         adamw_params=other_params_disc,
    #         adamw_lr=lr,
    #         adamw_betas=(0.90, 0.95),
    #         adamw_wd=0.01
    #     )

    #     # Return both optimizers
    #     return [opt_ae, opt_disc], []

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


# class VQMultiModel1(pl.LightningModule):
#     def __init__(self,
#                  ddconfig={
#                      "double_z": False,
#                      "ch": 128,
#                      "num_res_blocks": 2,
#                      "dropout": 0.0
#                  },
#                  lossconfig={
#                      "disc_conditional": False,
#                      "disc_in_channels": 3,
#                      "disc_start": 50000,
#                      "disc_weight": 0.4,
#                      "codebook_weight": 1.0
#                  },
#                  n_embed=8192,
#                  embed_dim=8,
#                  dropout_step=20000,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  remap=None,
#                  sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__()
#         self.image_key = image_key
#         self.automatic_optimization = False
        
#         z_channels = 256
        
#         self.encoder_1 = Encoder(**ddconfig, ch_mult=[1,1,2,2], in_channels=3, out_ch=3, z_channels=z_channels, resolution=256, attn_resolutions=[32])
#         self.encoder_2 = Encoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=32, attn_resolutions=[16])
#         self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[4])
#         self.encoder_3 = Encoder(**ddconfig, ch_mult=[1,1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[4])
        
#         self.decoder_1 = Decoder(**ddconfig, ch_mult=[1,1,2,2], in_channels=712, out_ch=3, z_channels=z_channels * 3, resolution=256, attn_resolutions=[64])
#         self.decoder_2 = Decoder(**ddconfig, ch_mult=[1,2], in_channels=256, out_ch=256, z_channels=z_channels * 2, resolution=32, attn_resolutions=[16])
#         self.decoder_3 = Decoder(**ddconfig, ch_mult=[1,1,2], in_channels=256, out_ch=256, z_channels=z_channels, resolution=16, attn_resolutions=[4])
        
#         self.quantize_1 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap, sane_index_shape=sane_index_shape)
#         self.quantize_2 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap, sane_index_shape=sane_index_shape)
#         self.quantize_3 = VectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap, sane_index_shape=sane_index_shape)
        
#         self.quant_conv_1 = torch.nn.Conv2d(z_channels * 3, embed_dim, 1)
#         self.quant_conv_2 = torch.nn.Conv2d(z_channels * 2, embed_dim, 1)
#         self.quant_conv_3 = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
#         self.post_quant_conv_1 = torch.nn.Conv2d(embed_dim, z_channels, 1)
#         self.post_quant_conv_2 = torch.nn.Conv2d(embed_dim, z_channels, 1)
#         self.post_quant_conv_3 = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
#         self.dropout_step = dropout_step
        
#         self.loss = VQLPIPSWithDiscriminator(**lossconfig)
#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
#         self.image_key = image_key
#         if colorize_nlabels is not None:
#             assert type(colorize_nlabels)==int
#             self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
#         if monitor is not None:
#             self.monitor = monitor

#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path}")

#     def encode(self, x):
#         enc_1 = self.encoder_1(x)
#         enc_2 = self.encoder_2(enc_1)
#         enc_3 = self.encoder_3(enc_2)
        
#         conv_3 = self.quant_conv_3(enc_3)
#         quant_3, emb_loss_3, info_3 = self.quantize_3(conv_3)
#         post_quant_3 = self.post_quant_conv_3(quant_3)
        
#         dec_3 = self.decoder_3(post_quant_3)
#         enc_2 = torch.cat([dec_3, enc_2], dim=1)
        
#         conv_2 = self.quant_conv_2(enc_2)
#         quant_2, emb_loss_2, info_2 = self.quantize_2(conv_2)
#         post_quant_2 = self.post_quant_conv_2(quant_2)
        
#         dec_2 = self.decoder_2(torch.cat([dec_3, post_quant_2], dim=1))
#         enc_1 = torch.cat([dec_3.repeat(1, 1, 2, 2), dec_2, enc_1], dim=1)
        
#         conv_1 = self.quant_conv_1(enc_1)
#         quant_1, emb_loss_1, info_1 = self.quantize_1(conv_1)
        
#         return quant_1, quant_2, quant_3, emb_loss_1 + emb_loss_2 + emb_loss_3, (info_1, info_2, info_3)

#     def decode(self, quant1, quant2, quant3):
        
#         quant2 = quant2.repeat(1, 1, 2, 2)
#         quant3 = quant3.repeat(1, 1, 8, 8)
        
#         quant1 = self.post_quant_conv_1(quant1)
#         quant2 = self.post_quant_conv_2(quant2)
#         quant3 = self.post_quant_conv_3(quant3)
#         return self.decoder_1(torch.cat([quant1, quant2, quant3], dim=1))
    
#     def forward(self, input):
#         quant1, quant2, quant3, diff, _ = self.encode(input)
#         b=input.shape[0]

#         if self.global_step > self.dropout_step:
#             with torch.no_grad():       
#                 zero_1 = self.quantize_1.get_codebook_entry(torch.zeros(b, dtype=torch.long, device=input.device), shape=(b, 1, 1, 8))
#                 zero_2 = self.quantize_2.get_codebook_entry(torch.zeros(b, dtype=torch.long, device=input.device), shape=(b, 1, 1, 8))
#                 # Calculate number of patches to zero out (0-50% randomly) for quant1
#                 h1, w1 = quant1.shape[2:]
#                 num_patches1 = h1 * w1
#                 num_zeros1 = torch.randint(0, num_patches1//2 + 1, (1,)).item()
                
#                 # Randomly select patches to zero out for quant1
#                 zero_indices1 = torch.randperm(num_patches1)[:num_zeros1]
                
#                 # Convert to 2D indices for quant1
#                 zero_h1 = (zero_indices1 // w1).long()
#                 zero_w1 = (zero_indices1 % w1).long()
                
#                 # Calculate number of patches to zero out (0-25% randomly) for quant2
#                 h2, w2 = quant2.shape[2:]
#                 num_patches2 = h2 * w2
#                 num_zeros2 = torch.randint(0, num_patches2//4 + 1, (1,)).item()
                
#                 # Randomly select patches to zero out for quant2
#                 zero_indices2 = torch.randperm(num_patches2)[:num_zeros2]
                
#                 # Convert to 2D indices for quant2
#                 zero_h2 = (zero_indices2 // w2).long()
#                 zero_w2 = (zero_indices2 % w2).long()
                
#                 # Replace selected patches with zero tokens
#                 for i in range(num_zeros1):
#                     quant1[:, :, zero_h1[i], zero_w1[i]] = zero_1.squeeze(-1).squeeze(-1)
#                 for i in range(num_zeros2):
#                     quant2[:, :, zero_h2[i], zero_w2[i]] = zero_2.squeeze(-1).squeeze(-1)
        
#         dec = self.decode(quant1, quant2, quant3)
#         return dec, diff

#     def get_input(self, batch, k):
#         x = batch[k]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
#         return x.float()

#     def training_step(self, batch, batch_idx):
#         # Get optimizers
#         opt_ae, opt_disc = self.optimizers()
        
#         # Get input
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)

#         # First optimizer - autoencoder
#         opt_ae.zero_grad()
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
#                                         last_layer=self.get_last_layer(), split="train")
#         self.manual_backward(aeloss)
#         opt_ae.step()

#         # Second optimizer - discriminator
#         opt_disc.zero_grad()
#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
#                                         last_layer=self.get_last_layer(), split="train")
#         self.manual_backward(discloss)
#         opt_disc.step()

#         # Logging
#         self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
#         self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
#         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#         self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

#     def validation_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
#                                         last_layer=self.get_last_layer(), split="val")

#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
#                                             last_layer=self.get_last_layer(), split="val")
#         rec_loss = log_dict_ae["val/rec_loss"]
        
#         # Log metrics only once
#         self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
#         self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
#         # Log dictionaries without duplicate metrics
#         for key, value in log_dict_ae.items():
#             if key != "val/rec_loss":  # Skip already logged metrics
#                 self.log(key, value)
                
#         self.log_dict(log_dict_disc)
#         return self.log_dict

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         # List all parameters for the autoencoder optimizer:
#         # 1. Encoder/decoder networks
#         encoder_decoder_params = list(self.encoder_1.parameters()) + list(self.encoder_2.parameters()) + list(self.encoder_3.parameters()) + list(self.decoder.parameters())
        
#         # 2. Quantization components
#         quant_params = list(self.quantize_1.parameters()) + list(self.quantize_2.parameters()) + list(self.quantize_3.parameters())
        
#         # 3. Pre/post quantization convolutions
#         conv_params = list(self.quant_conv_1.parameters()) + list(self.quant_conv_2.parameters()) + list(self.quant_conv_3.parameters()) + list(self.post_quant_conv_1.parameters()) + list(self.post_quant_conv_2.parameters()) + list(self.post_quant_conv_3.parameters())
        
#         # Combine all parameters
#         opt_ae = torch.optim.Adam(
#             encoder_decoder_params + quant_params + conv_params,
#             lr=lr, betas=(0.9, 0.95)
#         )
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr, betas=(0.9, 0.95))
        
#         sched_ae = lr_scheduler.LinearLR(opt_ae, start_factor=1.0, end_factor=0.05, total_iters=50)
#         sched_disc = lr_scheduler.LinearLR(opt_disc, start_factor=1.0, end_factor=0.05, total_iters=50)
        
#         return [opt_ae, opt_disc], [sched_ae, sched_disc]

#     def get_last_layer(self):
#         return self.decoder_1.conv_out.weight

#     def log_images(self, batch, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         xrec, _ = self(x)
#         if x.shape[1] > 3:
#             # colorize with random projection
#             assert xrec.shape[1] > 3
#             x = self.to_rgb(x)
#             xrec = self.to_rgb(xrec)
#         log["inputs"] = x
#         log["reconstructions"] = xrec
#         return log

#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.conv2d(x, weight=self.colorize)
#         x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
#         return x
    
        
#     def on_train_epoch_end(self, *args, **kwargs):
#         lr_ae, lr_disc = self.lr_schedulers()
#         lr_ae.step()
#         lr_disc.step()


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