import torch
from torch import nn

from models.state_encoder import StateEncoder


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=2, num_frames=20):
        super().__init__()
        self._state_encoder = StateEncoder(mode='encoder')
        self._encode_state = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 2),
        )

    def forward(self, x, mask=None, device=None):
        state = self._state_encoder(x, mask=mask, device=device)
        latent_statistics = self._encode_state(state)
        return torch.chunk(latent_statistics, 2, dim=-1)
