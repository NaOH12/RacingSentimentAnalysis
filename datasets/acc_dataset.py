import os

import numpy as np
from torch.utils.data import Dataset


class ACCDataset(Dataset):
    def __init__(self, *args, **kwargs):
        # Number of frames to predict
        self._future_frames = kwargs.get('future_frames', 15)
        # Limit the number of racing line points and cars to consider
        self._max_distance = kwargs.get('max_distance', 75)
        self._max_square_distance = self._max_distance ** 2

        self._data_dir = kwargs.get('data_dir', 'C:\\Users\\noahl\Documents\ACCDataset')
        self._sample_dir = os.path.join(self._data_dir, 'race_data')
        self._track_dir = os.path.join(self._data_dir, 'track_data')
        self._file_names = os.listdir(self._data_dir)
        self._file_ids = [file[:-4] for file in self._file_names]

        # Load the track data e.g. racing lines, track points...
        track_file_names = os.listdir(self._track_dir)
        self._track_data = {
            np.load(f"{self._track_dir}{file}", allow_pickle=True).item()['track_id']: file for file in track_file_names
        }

        # Load the file_meta
        self._file_meta = np.load(os.path.join(self._data_dir, 'file_meta.npy'), allow_pickle=True).item()

    def __len__(self):
        return sum([
            file_meta['file_num_samples'] for file_id, file_meta in self._file_meta.items()
        ])

    def __getitem__(self, idx):
        # Here idx is enough to select the data file,
        # but we wish to further sample a start frame within the file later.

        # Get the file_id from the idx
        count = 0
        for file_id, file_meta in self._file_meta.items():
            count+= file_meta['file_num_samples']
            if idx < count:
                idx = idx - count
                break

        # Load the file
        data = np.load(f"{self._sample_dir}{file_id}.npy", allow_pickle=True).item()
        sample_data = data['data']
        invalids = data['invalids']
        # Track data
        track_id = data['track_id']
        track_data = np.load(f"{self._track_dir}{self._track_data[track_id]}", allow_pickle=True).item()
        left_track = track_data['left_track']
        right_track = track_data['right_track']
        racing_line = track_data['racing_line']

        # TODO Construct probability map for the data based on car proximities.

        # Sample the car
        car_idx = np.random.choice(data.shape[0])
        frame_idx = idx

        sample_data = sample_data[frame_idx:frame_idx+self._future_frames]
        invalids = invalids[frame_idx:frame_idx+self._future_frames]

        # Previous frame
        prev_frame = sample_data[frame_idx-2]
        prev_invalids = invalids[frame_idx-2]

        # Get initial direction
        direction = sample_data[frame_idx] - prev_frame

        # TODO Assert invalids

        # Center translation
        sample_data = sample_data - sample_data[0:1, car_idx:car_idx+1]
        # Rotate the data based on focus car direction
        focus_car_direction = sample_data[car_idx]
        angle = np.arctan2(focus_car_direction[2], focus_car_direction[0])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        sample_data = np.dot(sample_data, rotation_matrix)
        racing_line = np.dot(racing_line, rotation_matrix)
        left_track = np.dot(left_track, rotation_matrix)
        right_track = np.dot(right_track, rotation_matrix)

