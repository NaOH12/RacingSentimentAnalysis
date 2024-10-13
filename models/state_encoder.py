import numpy as np
import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    This module exists outside of the VAE, and is to be used both by the encoder and decoder.
    It encodes the states of the cars, aggregates the car and racing line information.
    """
    def __init__(self, *args, hidden_dim=32, mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['encoder', 'decoder'], 'Mode must be either encoder or decoder'
        self._mode = mode
        self._hidden_dim = hidden_dim
        # Car state encoder
        if self._mode == 'encoder':
            # Encoder mode
            self._car_encoder = nn.Sequential(
                nn.Conv1d(in_channels=5*3, out_channels=self._hidden_dim // 4, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=self._hidden_dim // 4, out_channels=self._hidden_dim // 4, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=self._hidden_dim // 4, out_channels=self._hidden_dim // 2, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=self._hidden_dim // 2, out_channels=self._hidden_dim // 2, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=self._hidden_dim // 2, out_channels=self._hidden_dim // 2, kernel_size=3, stride=2),
                nn.ReLU()
            )
        else:
            # Decoder mode
            self._car_encoder = nn.Sequential(
                nn.Conv1d(in_channels=5 * 3, out_channels=self._hidden_dim // 4, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self._hidden_dim // 4, out_channels=self._hidden_dim // 2, kernel_size=1, stride=1),
                nn.ReLU()
            )
        # Track point encoder
        self._border_encoder = nn.Sequential(
            nn.Linear(3, self._hidden_dim // 2),
            nn.ReLU(),
        )
        self._racing_line_encoder = nn.Sequential(
            nn.Linear(3, self._hidden_dim // 2),
            nn.ReLU(),
        )
        # There are variable number of cars and track points...
        self._car_track_transformer = nn.Transformer(
            nhead=4, num_encoder_layers=1, num_decoder_layers=1, batch_first=True,
            d_model=self._hidden_dim // 2, dim_feedforward=self._hidden_dim
        )


    def forward(self, x, embeddings=None, mask=None, device=None):
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

        # Encode the car positions to get motion segments of the sequence
        # Car pos shape (bs, frames, num_cars, 5, 3) -> (bs * num_cars, 5*3, frames)
        car_positions = sample_data.permute(0, 2, 3, 4, 1).reshape(bs * num_cars, 5*3, frames)
        car_conv_encodings = self._car_encoder(car_positions)
        # Car pos shape (bs * num_cars, hidden_dim, segments) -> (bs, segments, num_cars, hidden_dim)
        car_conv_encodings = car_conv_encodings.reshape(bs, num_cars, self._hidden_dim, -1).permute(0, 3, 1, 2)

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

        num_segments = car_conv_encodings.shape[1]
        states = []
        # For each segment
        for i in range(num_segments):
            car_encodings = car_conv_encodings[:, i]

            if embeddings is not None:
                car_encodings = car_encodings + embeddings

            # Apply car-track transformer
            state = self._car_track_transformer(
                track_encodings, car_encodings,
                src_key_padding_mask=~track_mask, tgt_key_padding_mask=~car_mask
            )
            states.append(state)

        state = torch.stack(states, dim=1).mean(dim=1)

        return state
