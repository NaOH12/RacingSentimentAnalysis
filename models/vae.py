import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.acc_dataset import ACCDataset
from models.state_encoder import StateEncoder


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._encode_state = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, state):
        return self._encode_state(state)

class Decoder(nn.Module):
    def __init__(self, hidden_state=32, num_frames_to_predict=19):
        super().__init__()
        self._hidden_state = hidden_state
        self._num_frames_to_predict = num_frames_to_predict
        self._decode_latent = nn.Sequential(
            nn.Linear(4, self._hidden_state),
            nn.ReLU(),
        )
        self._combine_decode = nn.Sequential(
            nn.Linear(64 + 32, self._hidden_state),
            nn.ReLU(),
            nn.Linear(self._hidden_state, self._num_frames_to_predict * 3),
        )

    def forward(self, state, latent):
        decoded_latent = self._decode_latent(latent)
        combined = torch.cat([state, decoded_latent], dim=-1)
        return self._combine_decode(combined)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self._state_encoder = StateEncoder()
        self._encoder = Encoder()
        self._decoder = Decoder()

    def forward(self, x):
        state = self._state_encoder(x)
        latent = self._encoder(state)
        reconstruct = self._decoder(state, latent)
        reconstruct = reconstruct.view(-1, 19, 3)
        return reconstruct


if __name__ == '__main__':
    # Test the ACCDataset
    dataset = ACCDataset()
    # Create a dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for x, y in loader:
        response = VAE()(x)
        break

