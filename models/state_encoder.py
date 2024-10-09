import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    This module exists outside of the VAE, and is to be used both by the encoder and decoder.
    It encodes the states of the cars, aggregates the car and racing line information.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Input feature size per car ((4 wheels + 1 center coords) + velocity)
        self._car_feat_dim = 5*3 + 3
        self._hidden_dim = 32
        self._out_dim = 64
        # Car state encoder
        self._car_encoder = nn.Sequential(
            nn.Linear(self._car_feat_dim, self._hidden_dim),
            nn.ReLU(),
        )
        # Track point encoder
        self._track_encoder = nn.Sequential(
            nn.Linear(3, self._hidden_dim),
            nn.ReLU(),
        )
        # Learnable embedding
        self._focus_embedding = nn.Parameter(torch.randn(1, self._hidden_dim))
        self._non_focus_embedding = nn.Parameter(torch.randn(1, self._hidden_dim))

        # There are variable number of cars and track points...
        self._car_track_transformer = nn.Transformer(
            nhead=2, num_encoder_layers=2, num_decoder_layers=2, batch_first=True, d_model=self._hidden_dim
        )

        # Here we wish to use self attention on the cars
        self._car_self_attn = nn.MultiheadAttention(
            embed_dim=self._hidden_dim, num_heads=8
        )
        self._layer_norm = nn.LayerNorm(self._hidden_dim)

        self._linear_out = nn.Sequential(
            nn.Linear(self._hidden_dim, self._out_dim),
            nn.ReLU(),
        )


    def forward(self, x):          # (bs, 1)
        directions = x['directions']                # (bs, num_cars, 3)
        car_positions = x['car_positions'][:, 0]    # (bs, num_cars, 5, 3)
        racing_line = x['racing_line']              # (bs, num_points, 3)

        bs, num_cars, in_c, _ = car_positions.shape

        # Prepare data
        # (bs, num_cars, 5, 3) -> (bs, num_cars, 5 * 3)
        car_positions = car_positions.reshape(bs, num_cars, -1)
        car_data = torch.concat([car_positions, directions], dim=-1)

        # Encode the car positions and racing line points
        car_encodings = self._car_encoder(car_data)
        track_encodings = self._track_encoder(racing_line)

        # Apply the embeddings (this is probably not necessary, since the focus car is always at origin)
        car_encodings[:, 0] += self._focus_embedding
        car_encodings[:, 1:] += self._non_focus_embedding[:, None]

        # Apply car-track transformer
        car_track_encodings = self._car_track_transformer(
            track_encodings, car_encodings
        )

        # Perform self attention amongst the cars
        attn_input = self._layer_norm(car_track_encodings)
        state = car_track_encodings + self._car_self_attn(attn_input, attn_input, attn_input)[0]
        state = self._linear_out(state[:, 0])

        return state
