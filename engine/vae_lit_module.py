from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from models.vae import VAE


class LitVAEModule(L.LightningModule):
    def __init__(
            self, num_frames, latent_dim
    ):
        super().__init__()
        self.model = VAE(num_frames=num_frames, latent_dim=latent_dim)

    def step(self, x):
        mu, log_var, latent, reconstruct = self.model(x)
        target = x['sample_data'][:, :, :, 0, :]

        # VAE loss with regularization
        recon_loss = torch.nn.functional.mse_loss(reconstruct, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss * 0.002 + kl_loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
