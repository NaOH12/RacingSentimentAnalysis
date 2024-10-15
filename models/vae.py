import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.acc_dataset import ACCDataset
from models.state_encoder import StateEncoder
from models.vae_decoder import VAEDecoder
from models.vae_encoder import VAEEncoder


class VAE(nn.Module):
    def __init__(self, num_frames, latent_dim, hidden_dim=None, skip_frames=None):
        super().__init__()
        self._encoder = VAEEncoder(num_frames=num_frames, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self._decoder = VAEDecoder(
            num_frames=num_frames, latent_dim=latent_dim,
            hidden_dim=hidden_dim, skip_frames=skip_frames
        )

    def _construct_mask(self, x, device=None):
        # Since we have variable number of cars and track points we need
        # to mask the attention mechanism
        car_mask = torch.ones_like(x['sample_data'][:, 0, :, 0, 0], device=device).bool()
        # car_mask_square = torch.zeros(car_mask.shape[0], car_mask.shape[1], car_mask.shape[1])
        for i, num_cars in enumerate(x['num_cars']):
            car_mask[i, num_cars:] = False
            # car_mask_square[i, num_cars:, num_cars:] = -torch.inf

        num_border_points = x['num_border_points']
        num_racing_line_points = x['num_racing_line_points']
        border_points = x['border_points']
        racing_line_points = x['racing_line_points']
        border_points_mask = x['border_points_mask']
        racing_line_points_mask = x['racing_line_points_mask']

        bs = x['sample_data'].shape[0]
        track_mask = torch.ones((bs, border_points.shape[1] + racing_line_points.shape[1]), device=device).bool()
        for i in range(bs):
            track_mask[i, (x['num_border_points'][i] + x['num_racing_line_points'][i]):] = False

        return {
            'car_mask': car_mask,
            # 'car_mask_square': car_mask_square,
            'track_mask': track_mask,
            # 'track_mask_square': track_mask_square
        }

    def forward(self, x):
        device = x['sample_data'].device

        mask = self._construct_mask(x, device=device)

        mu, log_var = self._encoder(x, mask=mask, device=device)

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            latent = mu + eps * std
        else:
            latent = mu

        reconstruct = self._decoder(x, latent, mask=mask, device=device)

        # # Each previous prediction is added to the next prediction
        # for i in range(1, reconstruct.shape[1]):
        #     reconstruct[:, i] += reconstruct[:, i-1]
        # Add reconstruct to x
        # reconstruct += x['sample_data'][:, 0:1, :, 0, :]

        return mu, log_var, latent, reconstruct


if __name__ == '__main__':
    num_frames=50
    # Test the ACCDataset
    dataset = ACCDataset(num_frames=num_frames)
    # Create a dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for x in loader:
        # Convert items in x to cuda
        for key in x:
            x[key] = x[key].cuda()
        if x['sample_data'].shape[2] <= 1:
            continue
        break

    model = VAE(num_frames=num_frames, latent_dim=2).cuda()

    # Get param count
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    response = model(x)
