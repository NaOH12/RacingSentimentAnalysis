import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.acc_dataset import ACCDataset
from models.state_encoder import StateEncoder
from models.vae_decoder import VAEDecoder
from models.vae_encoder import VAEEncoder


class VAE(nn.Module):
    def __init__(self, num_frames, latent_dim):
        super().__init__()
        self._encoder = VAEEncoder(num_frames=num_frames, latent_dim=latent_dim)
        self._decoder = VAEDecoder(num_frames=num_frames, latent_dim=latent_dim)

    def _construct_mask(self, x):
        # Since we have variable number of cars and track points we need
        # to mask the attention mechanism
        car_mask = torch.zeros_like(x['sample_data'][:, 0, :, 0, 0]).bool()
        car_mask_square = torch.zeros(car_mask.shape[0], car_mask.shape[1], car_mask.shape[1])
        for i, num_cars in enumerate(x['num_cars']):
            car_mask[i, num_cars:] = True
            car_mask_square[i, num_cars:, num_cars:] = -torch.inf

        track_mask = torch.zeros_like(x['racing_line'][:, :, 0]).bool()
        track_mask_square = torch.zeros(track_mask.shape[0], track_mask.shape[1], track_mask.shape[1])
        for i, num_points in enumerate(x['num_racing_line_points']):
            track_mask[i, num_points:] = True
            track_mask_square[i, num_points:, num_points:] = -torch.inf

        return {
            'car_mask': car_mask,
            'car_mask_square': car_mask_square,
            'track_mask': track_mask,
            'track_mask_square': track_mask_square
        }

    def forward(self, x, device=None):
        mask = self._construct_mask(x)

        mu, log_var = self._encoder(x, mask=mask)

        if self.training:
            latent = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        else:
            latent = mu

        reconstruct = self._decoder(x, latent, mask=mask)
        return mu, log_var, latent, reconstruct


if __name__ == '__main__':
    num_frames=50
    # Test the ACCDataset
    dataset = ACCDataset(num_frames=num_frames)
    # Create a dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for x in loader:
        if x['sample_data'].shape[2] <= 1:
            continue
        break

    model = VAE(num_frames=num_frames, latent_dim=2)
    model.train()

    # Get param count
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    response = model(x)