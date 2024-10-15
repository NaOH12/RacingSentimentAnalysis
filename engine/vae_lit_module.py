from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from models.vae import VAE
from visualize.data_player import ReplayPlayer


class LitVAEModule(L.LightningModule):
    def __init__(
            self, num_frames, latent_dim, skip_frames=10
    ):
        super().__init__()
        self._skip_frames = skip_frames
        self.model = VAE(num_frames=num_frames, latent_dim=latent_dim, hidden_dim=32, skip_frames=skip_frames)

    def step(self, x):
        mu, log_var, latent, reconstruct = self.model(x)
        # target = x['sample_data'][:, :, :, 0, :]
        target = x['sample_data'][:, :, :, 0, :]
        num_cars = x['num_cars'][0]

        # VAE loss with regularization
        recon_loss = torch.nn.functional.mse_loss(reconstruct[:, :, :num_cars], target[:, ::self._skip_frames, :num_cars])
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Visualize the reconstruction
        if self.training and self.global_step % 5 == 0:
            num_cars = x['num_cars'][0]
            vis_cars, vis_border_points, vis_racing_line_points, vis_reconstruct = (
                x['sample_data'][0, x['start_skip_frames'][0]:x['start_skip_frames'][0]+1, :num_cars].cpu().numpy(),
                x['border_points'][0].cpu().numpy(),
                x['racing_line_points'][0].cpu().numpy(),
                reconstruct[0:1, :, :].detach().cpu().numpy()
            )
            # player = ReplayPlayer(
            #     x['sample_data'][0, 0:1, :num_cars].cpu().numpy(),
            #     x['border_points'][0].cpu().numpy(), x['racing_line_points'][0].cpu().numpy(),
            #     preds=reconstruct[0:1, :, :num_cars].detach().cpu().numpy()
            # )
            # out = player.render(car_focus_id=0, to_numpy=True, window_size=(500, 500))[0]
            # import matplotlib.pyplot as plt
            # # Fullscreen plot
            # fig = plt.figure()
            # plt.imshow(out)
            # plt.show()
            # plt.close()

            # Visualize reconstruct
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            # Plot cars
            plt.scatter(vis_cars[0, :, 0, 0], vis_cars[0, :, 0, 2], c='blue', label='Car')
            # Plot border points
            plt.scatter(vis_border_points[:, 0], vis_border_points[:, 2], c='purple', label='Border', s=0.5)
            # Plot racing line points
            # plt.scatter(vis_racing_line_points[:, 0], vis_racing_line_points[:, 2], c='green', label='Racing Line', s=0.5)
            # Visualize ground truth
            for i in range(num_cars):
                plt.plot(target[0, :, i, 0].cpu(), target[0, :, i, 2].cpu(), c='green', alpha=0.5)
                plt.plot(vis_reconstruct[0, :, i, 0], vis_reconstruct[0, :, i, 2], c='red', alpha=0.5)
            plt.show()
            plt.close()

        beta = min((self.global_step / 500), 1.0)
        return (recon_loss + beta * 0.002 * kl_loss) * 1000

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        # Cycle scheduler
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=1e-3,
        #     total_steps=self.trainer.estimated_stepping_batches
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        # return optimizer
        return [optimizer], [scheduler]