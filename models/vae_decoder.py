import numpy as np
import torch
from sympy.physics.units import acceleration
from torch import nn
from torch.xpu import device

from models.state_encoder import StateEncoder


class VAEDecoder(nn.Module):
    def __init__(self, hidden_dim=None, num_frames=None, latent_dim=2, skip_frames=None, accel_scale=0.01):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_frames = num_frames
        self._skip_frames = skip_frames
        self._accel_scale = accel_scale
        # We decode the latent, ideally it will represent some behaviour of the car.
        self._decode_latent = nn.Sequential(
            nn.Linear(latent_dim, self._hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.LeakyReLU(),
        )
        self._state_encoder = StateEncoder(mode='decoder', hidden_dim=self._hidden_dim)
        self._state_decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self._hidden_dim, 3),
            nn.Tanh()
        )
        # self._self_attn = nn.MultiheadAttention(
        #     embed_dim=self._hidden_dim, num_heads=4
        # )
        # self._linear_out = nn.Sequential(
        #     nn.Linear(self._hidden_dim, 3),
        # )
        # self._layer_norm = nn.LayerNorm(self._hidden_dim)
        # self._simple_decoder = nn.Sequential(
        #     nn.Linear(6 + 32, self._hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self._hidden_dim, self._hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self._hidden_dim, 3),
        #     nn.Sigmoid()
        # )

    def forward(self, x, latent, mask=None, device=None):
        # new_x = {**x}
        # new_x['sample_data'] = x['sample_data'][:, :x['start_skip_frames'][0]+1]

        # Decoded latent representation
        decoded_latent = self._decode_latent(latent)
        decoded_latent = torch.ones_like(decoded_latent)

        # If training, then for each frame we predict the acceleration
        # from the previous ground truth... Otherwise, training is very unstable!
        if self.training is True:
            # Num frames
            num_reduced_frames = self._num_frames // self._skip_frames
            # Skip every other frame
            sample_data = x['sample_data'][:, ::self._skip_frames]
            # Calculate the velocities for each frame (for 1:)
            velocities = sample_data[:, 1:, :, 0] - sample_data[:, 0:-1, :, 0]
            # # Get the predicted acceleration
            # acceleration = (self._simple_decoder(torch.concat([
            #     # velocities, x['sample_data'][:, 1:, :, 0], one_hot.expand((1, 99, 64, 64)),
            #     velocities, sample_data[:, 1:, :, 0], state[:, None].expand((-1, num_reduced_frames - 1, -1, -1)),
            # ], dim=-1)) / 100) * self._skip_frames

            bs, _, num_cars, in_c, _ = sample_data.shape
            acceleration = [
                self._state_decoder(self._state_encoder(
                    mask=mask, device=device, **{**x, "sample_data": x['sample_data'][:, i:i+1, :, 0].reshape(bs, 1, num_cars, -1)},
                    velocity=(x['sample_data'][:, i:i+1, :, 0] - x['sample_data'][:, i-1:i, :, 0]).reshape(bs, 1, num_cars, -1),
                    behaviour_embeddings=decoded_latent[:, None, :, :].reshape(bs, 1, num_cars, self._hidden_dim)
                ))
                for i in range(1, num_reduced_frames - 1)
            ]
            acceleration = torch.stack(acceleration, dim=1) * self._accel_scale * self._skip_frames

            outputs = torch.concat([
                sample_data[:, 0:2, :, 0],
                # Position [1:] + Velocity at [1:] + Pred. Acceleration at [1:]
                sample_data[:, 1:-1, :, 0] + velocities[:, :-1] + acceleration[:, :],
            ], axis=1)
        else:
            # At inference, we run in an autoregressive manner
            num_reduced_frames = self._num_frames // self._skip_frames
            sample_data = x['sample_data'][:, ::self._skip_frames]
            # Velocity shape (bs, 1, num_cars, 3)
            velocities = sample_data[:, 1:2, :, 0] - sample_data[:, 0:1, :, 0]
            outputs = [x['sample_data'][:, 0:1, :, 0, :]]

            bs, _, num_cars, in_c, _ = sample_data.shape
            # # Car wheel contact coords of shape (bs, 1, num_cars, 4, 3)
            # contact_point_pos = x['sample_data'][:, 0:1, :, 1:] - x['sample_data'][:, 0:1, :, 0:1]
            # # Car unit initial direction (bs, 1, num_cars, 3)
            # car_direction = velocities[..., :] / torch.norm(velocities[..., :], dim=-1, keepdim=True)

            for i in range(1, num_reduced_frames - 1):
                acceleration = self._state_decoder(self._state_encoder(
                    mask=mask, device=device, **{**x, "sample_data": outputs[-1].reshape(bs, 1, num_cars, -1)},
                    velocity=velocities,
                    behaviour_embeddings=decoded_latent[:, None, :, :].reshape(bs, 1, num_cars, self._hidden_dim)
                ))
                acceleration *= self._accel_scale * self._skip_frames
                velocities = velocities + acceleration[:, None, ...]
                outputs.append(outputs[-1].reshape(bs, 1, num_cars, -1) + velocities)
            outputs = torch.concat(outputs, dim=1)

        return outputs
