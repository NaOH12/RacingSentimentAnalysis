import numpy as np
import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    This module exists outside of the VAE, and is to be used both by the encoder and decoder.
    It encodes the states of the cars, aggregates the car and racing line information.
    """
    def __init__(self, *args, num_frames=20, hidden_dim=32, out_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        # Input feature size per car ((4 wheels + 1 center coords) + velocity)
        self._car_feat_dim = num_frames*5*3 + 3
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        # Car state encoder
        self._car_encoder = nn.Sequential(
            nn.Linear(self._car_feat_dim, self._hidden_dim),
            nn.ReLU(),
        )
        # Track point encoder
        self._border_encoder = nn.Sequential(
            nn.Linear(3, self._hidden_dim),
            nn.ReLU(),
        )
        self._racing_line_encoder = nn.Sequential(
            nn.Linear(3, self._hidden_dim),
            nn.ReLU(),
        )

        # There are variable number of cars and track points...
        self._car_track_transformer = nn.Transformer(
            nhead=4, num_encoder_layers=1, num_decoder_layers=1, batch_first=True, d_model=self._hidden_dim, dim_feedforward=self._hidden_dim
        )

        # Here we wish to use self attention on the cars
        self._car_self_attn = nn.MultiheadAttention(
            embed_dim=self._hidden_dim, num_heads=4, batch_first=True
        )

        self._linear_out = nn.Sequential(
            nn.Linear(self._hidden_dim, self._out_dim),
            nn.ReLU(),
        )


    def forward(self, x, embeddings=None, mask=None, device=None):
        initial_velocity = x['initial_velocity']
        sample_data = x['sample_data']
        num_border_points = x['num_border_points']
        num_racing_line_points = x['num_racing_line_points']
        border_points = x['border_points']
        racing_line_points = x['racing_line_points']
        border_points_mask = x['border_points_mask']
        racing_line_points_mask = x['racing_line_points_mask']
        track_mask = mask['track_mask']
        car_mask = mask['car_mask']

        bs, frames, num_cars, in_c, _ = sample_data.shape

        # Prepare data
        # (bs, frames, num_cars, 5, 3) -> (bs, num_cars, frames * 5 * 3)
        car_positions = sample_data.permute(0, 2, 1, 3, 4).reshape(bs, num_cars, -1)
        car_data = torch.concat([initial_velocity, car_positions], dim=-1)

        # Encode the car positions
        car_encodings = self._car_encoder(car_data)

        # Encode the track points
        border_encodings = self._border_encoder(border_points)
        racing_line_encodings = self._racing_line_encoder(racing_line_points)

        # Combine the border and racing line encodings
        # However we have the problem that we don't have contiguous points therefore
        # masking the attention is more of a headache...
        non_contiguous_mask = torch.cat([border_points_mask, racing_line_points_mask], dim=1)
        non_contiguous_track_encodings = torch.cat([border_encodings, racing_line_encodings], dim=1)

        track_encodings = torch.zeros_like(non_contiguous_track_encodings)
        track_encodings[track_mask] = non_contiguous_track_encodings[non_contiguous_mask]

        if embeddings is not None:
            car_encodings = car_encodings + embeddings

        # Apply car-track transformer
        car_track_encodings = self._car_track_transformer(
            track_encodings, car_encodings,
            src_key_padding_mask=~track_mask, tgt_key_padding_mask=~car_mask
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(car_track_encodings[0].detach().numpy())
        # plt.show()

        # Perform self attention amongst the cars
        # attn_input = self._layer_norm(car_track_encodings)
        state = car_track_encodings + self._car_self_attn(
            car_track_encodings, car_track_encodings, car_track_encodings,
            key_padding_mask=~car_mask
        )[0]
        state = self._linear_out(state)

        return state
