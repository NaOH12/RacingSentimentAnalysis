import os

import numpy as np
import torch
from sympy.stats import sample
from sympy.stats.rv import probability
from torch.utils.data import Dataset, DataLoader


class ACCDataset(Dataset):
    def __init__(self, *args, **kwargs):
        # Number of future frames to predict
        self._future_frames = kwargs.get('num_frames', 50)
        # Skip the first few frames to get initial direction info
        self._start_skip_frames = kwargs.get('start_skip_frames', 2)
        # Limit the number of racing line points and cars to consider
        self._max_distance = kwargs.get('max_distance', 50)
        self._max_square_distance = self._max_distance ** 2
        self._max_border_points = kwargs.get('max_border_points', 2000)
        self._max_racing_line_points = kwargs.get('max_racing_line_points', 1000)

        self._data_dir = kwargs.get('data_dir', 'C:\\Users\\noahl\Documents\ACCDataset')
        self._sample_dir = os.path.join(self._data_dir, 'race_data')
        self._track_dir = os.path.join(self._data_dir, 'track_data')
        self._file_names = os.listdir(self._sample_dir)
        self._file_ids = [file[:-4] for file in self._file_names]

        # Load the track data e.g. racing lines, track points...
        track_file_names = os.listdir(self._track_dir)
        self._track_data = {
            int(file_name[:-4]): np.load(os.path.join(self._track_dir, file_name), allow_pickle=True).item() for file_name in track_file_names
        }

        # Load the file_meta
        self._file_meta = np.load(os.path.join(self._data_dir, 'file_meta.npy'), allow_pickle=True).item()

        self._max_cars = max([meta['file_num_cars'] for meta in self._file_meta.values()])

        # Cache the data if enough memory is available
        self._is_data_cached = kwargs.get('is_data_cached', True)
        self._data_cache = {}

    def __len__(self):
        return sum([
            file_meta['file_num_samples'] - self._future_frames - self._start_skip_frames
            for file_id, file_meta in self._file_meta.items()
            if file_meta['track_id'] <= 11
        ])

    def _load_file(self, file_id):
        if self._is_data_cached and file_id in self._data_cache:
            data = self._data_cache[file_id]
        else:
            data = np.load(os.path.join(self._sample_dir, f"{file_id}.npy"), allow_pickle=True).item()
            self._data_cache[file_id] = data
        return data

    def __getitem__(self, idx):
        # Here idx is enough to select the data file,
        # but we wish to further sample a start frame within the file later.

        # Get the file_id from the idx
        count = 0
        for file_id, file_meta in self._file_meta.items():
            if file_meta['track_id'] > 11:
                continue
            count+= file_meta['file_num_samples'] - self._future_frames - self._start_skip_frames
            if idx < count:
                break

        # Load the file
        data = np.load(os.path.join(self._sample_dir, f"{file_id}.npy"), allow_pickle=True).item()
        sample_data = data['data'].astype(np.float32)
        invalids = data['invalids'].astype(np.bool)
        # Track data
        track_id = data['track_id']
        track_data = self._track_data[track_id]
        border_points = np.concatenate(
            [track_data['left_track'], track_data['right_track']], axis=0
        ).astype(np.float32)
        racing_line_points = track_data['racing_line'].astype(np.float32)

        # Construct probability map for the data based on car proximity and invalids.
        # Invalid regions should not be considered for sampling.
        invalids[:self._start_skip_frames] = True
        invalids[-self._future_frames:] = True
        valid_frame_mask = ~invalids
        # Compute square distance between cars, we should sample more "interesting" sequences.
        car_square_distances = ((
            sample_data[:, None, :, 0] - sample_data[:, :, None, 0]
        ) ** 2).sum(-1)
        probability_map = np.zeros_like(sample_data[:, :, 0, 0])
        try:
            if sample_data.shape[1] == 1:
                probability_map[valid_frame_mask] = 1
                probability_map /= probability_map.sum()
            else:
                probability_map[valid_frame_mask] = (
                        1 - (car_square_distances[valid_frame_mask] / car_square_distances[valid_frame_mask].max())
                ).sum(-1)
                probability_map[valid_frame_mask] /= probability_map.sum()
                probability_map[valid_frame_mask] = np.exp(probability_map[valid_frame_mask])
                probability_map[valid_frame_mask] /= probability_map.sum()
            # Given the 2d probability map, sample both dimensions
            frame_idx, focus_car_idx = np.unravel_index(
                np.random.choice(probability_map.size, p=probability_map.ravel()), probability_map.shape
            )
        except Exception as e:
            pass

        # TODO ***

        # Filter data by computing square distances across the sequence, and selecting
        # data that exists within the min distance at least once in the episode.
        # Filter the track points
        border_points_sqr_distances = ((border_points[:, None] - sample_data[frame_idx:frame_idx+self._future_frames, focus_car_idx, 0][None, ...]) ** 2).sum(-1)
        border_points_sqr_distances = border_points_sqr_distances.min(-1)
        border_points = border_points[border_points_sqr_distances < self._max_square_distance]
        racing_line_sqr_distances = ((racing_line_points[:, None] - sample_data[frame_idx:frame_idx+self._future_frames, focus_car_idx, 0][None, ...]) ** 2).sum(-1)
        racing_line_sqr_distances = racing_line_sqr_distances.min(-1)
        racing_line_points = racing_line_points[racing_line_sqr_distances < self._max_square_distance]

        # racing_line_points_mask = track_points_sqr_distances[racing_line_start_idx:] < self._max_square_distance
        # track_points_mask = np.concatenate([border_points_mask, racing_line_points_mask], axis=0)
        # racing_line_start_idx = border_points_mask.nonzero()[0].max() + 1
        # track_points = track_points[track_points_mask]
        # Filter the number of cars
        car_filter = (
            car_square_distances[frame_idx:frame_idx+self._future_frames, focus_car_idx].min(0)
        ) < self._max_square_distance
        sample_data = sample_data[:, car_filter]
        invalids = invalids[:, car_filter]
        # Get new focus car index
        focus_car_idx = np.where(np.where(car_filter)[0] == focus_car_idx)[0][0]

        # Select the sequence data
        sample_data = sample_data[frame_idx-self._start_skip_frames:frame_idx+self._future_frames]
        invalids = invalids[frame_idx-self._start_skip_frames:frame_idx+self._future_frames]

        # Here we lose the notion of the "focus car".
        # Instead we will take the mean of the racing line points.
        # TODO Maybe revert this...
        mean_point = border_points.mean(0)

        border_points = border_points - mean_point[None, :]
        racing_line_points = racing_line_points - mean_point[None, :]
        sample_data = sample_data - mean_point[None, None, None, :]

        # Rotate the data based on focus car direction. Shape is (3,)
        # Focus car orientation
        focus_car_direction = sample_data[0, focus_car_idx, 1] - sample_data[0, focus_car_idx, 3]
        focus_car_direction /= np.linalg.norm(focus_car_direction)

        # Construct 3d rotation matrix
        # angle = np.arctan2(focus_car_direction[0], focus_car_direction[2])
        # Random angle TODO Maybe change this...
        angle = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]).astype(np.float32)

        # Sample data shape (n, n_cars, c, 3) -> (n * n_cars * c, 3)
        sample_data_shape = sample_data.shape
        sample_data = sample_data.reshape(-1, 3)

        # Rotate the data
        sample_data = np.matmul(sample_data, rotation_matrix)
        border_points = np.matmul(border_points, rotation_matrix)
        racing_line_points = np.matmul(racing_line_points, rotation_matrix)

        sample_data = sample_data.reshape(*sample_data_shape)

        # # Move the focus car to the first position
        # sample_data = np.roll(sample_data, -focus_car_idx, axis=1)

        # Pad the sample_data with zeros to get the maximum number of cars
        num_cars = sample_data.shape[1]
        sample_data = np.concatenate([
            sample_data,
            np.zeros((
                sample_data.shape[0], self._max_cars - sample_data.shape[1],
                sample_data.shape[2], sample_data.shape[3]
            ), dtype=np.float32)
        ], axis=1)
        # Pad the track points with zeros to get the maximum number of points
        num_border_points = border_points.shape[0]
        border_points = np.concatenate([
            border_points,
            np.zeros((
                self._max_border_points - num_border_points, 3),
                dtype=np.float32
            )
        ], axis=0)
        num_racing_line_points = racing_line_points.shape[0]
        racing_line_points = np.concatenate([
            racing_line_points,
            np.zeros((
                self._max_racing_line_points - num_racing_line_points, 3),
                dtype=np.float32
            )
        ], axis=0)
        # Create masks so we can treat these points separately.
        # Without the headache of extra transformers, pad masking etc.
        border_points_mask = np.zeros(self._max_border_points).astype(np.bool)
        border_points_mask[:num_border_points] = True
        racing_line_points_mask = np.zeros(self._max_racing_line_points).astype(np.bool)
        racing_line_points_mask[:num_racing_line_points] = True

        # Construct the training data
        return {
            'start_skip_frames': self._start_skip_frames,
            'num_cars': num_cars,
            'sample_data': sample_data,
            # Track information
            'num_border_points': num_border_points,
            'num_racing_line_points': num_racing_line_points,
            'border_points': border_points,
            'racing_line_points': racing_line_points,
            'border_points_mask': border_points_mask,
            'racing_line_points_mask': racing_line_points_mask,
        }


if __name__ == '__main__':
    num_frames=50
    # Test the ACCDataset
    dataset = ACCDataset(num_frames=num_frames)
    # Create a dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    max_count = 100
    for i, x in enumerate(loader):
        # if x['sample_data'].shape[2] <= 1:
        #     continue
        # break
        if i >= max_count:
            break
