import torch
from torch import nn

from models.state_encoder import StateEncoder


class VAEEncoder(nn.Module):
    def __init__(self, num_frames=None, latent_dim=2, hidden_dim=None):
        super().__init__()
        self._state_encoder = StateEncoder(mode='encoder', hidden_dim=hidden_dim)
        self._encode_state = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, latent_dim * 2),
        )

    def forward(self, x, mask=None, device=None):
        state = self._state_encoder(mask=mask, device=device, **x)
        latent_statistics = self._encode_state(state)
        return torch.chunk(latent_statistics, 2, dim=-1)
