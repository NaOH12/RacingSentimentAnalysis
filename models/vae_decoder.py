import torch
from torch import nn

from models.state_encoder import StateEncoder


class VAEDecoder(nn.Module):
    def __init__(self, hidden_dim=32, num_frames=20, latent_dim=2, skip_frames=0):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_frames = num_frames
        self._skip_frames = skip_frames
        # We decode the latent, ideally it will represent some behaviour of the car.
        self._decode_latent = nn.Sequential(
            nn.Linear(latent_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
        )
        self._state_encoder = StateEncoder(num_frames=1, hidden_dim=self._hidden_dim)
        # self._self_attn = nn.MultiheadAttention(
        #     embed_dim=self._hidden_dim, num_heads=4
        # )
        # self._linear_out = nn.Sequential(
        #     nn.Linear(self._hidden_dim, 3),
        # )
        # self._layer_norm = nn.LayerNorm(self._hidden_dim)
        self._simple_decoder = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._num_frames * 3),
        )

    def forward(self, x, latent, mask=None):
        new_x = {**x}
        new_x['sample_data'] = x['sample_data'][:, 0:1]
        # Decoded latent representation
        decoded_latent = self._decode_latent(latent)
        # Encode the track points, initial car positions and velocities
        state = self._state_encoder(new_x, embeddings=decoded_latent, mask=mask)
        # outputs = x['sample_data'][:, 0:1, :, 0, :]
        # # TODO use an autoregressive decoder to predict the future frames
        # # Encoder state is (bs, num_cars, dim)
        # # Decoder state is (bs, frames, num_cars, 3)
        # for i in range(1, self._num_frames):
        #     # Combine the latent with the state
        #     state = state + decoded_latent
        #     # Perform self attention
        #     attn_in = self._layer_norm(state)
        #     state = state + self._self_attn(attn_in, attn_in, attn_in)[0]
        #     outputs = torch.cat([outputs, self._linear_out(state[:, None])], dim=1)

        outputs = self._simple_decoder(state)
        outputs = outputs.reshape(-1, outputs.shape[1], self._num_frames, 3)

        # Set the output to zero for the cars that do not exist
        for i, num_cars in enumerate(x['num_cars']):
            outputs[i, num_cars:] = 0
        outputs = outputs.permute(0, 2, 1, 3)

        return outputs
