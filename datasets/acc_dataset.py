import os
import pickle
from os.path import split

import numpy as np
import torch
from sympy.stats import sample
from sympy.stats.rv import probability
from torch.utils.data import Dataset, DataLoader


class ACCDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(ACCDataset, self).__init__()
        self._split = kwargs.get('split', None)
        assert self._split in ['train', 'val', 'test'], "Invalid split"
        # Number of future frames to predict
        self._future_frames = kwargs.get('num_frames', 100)
        # Limit the number of racing line points and cars to consider
        self._max_border_points = kwargs.get('max_border_points', 2000)
        self._max_racing_line_points = kwargs.get('max_racing_line_points', 1000)

        self._data_dir = kwargs.get('data_dir', f'C:\\Users\\noahl\Documents\ACCDataset')
        self._sample_dir = os.path.join(self._data_dir, 'session_data', self._split)
        self._meta_dir = os.path.join(self._data_dir, 'meta_data', self._split)
        self._track_dir = os.path.join(self._data_dir, 'track_data')
        self._file_names = os.listdir(self._sample_dir)
        self._file_ids = [file[:-4] for file in self._file_names]

        # Load the track data e.g. racing lines, track points...
        track_file_names = os.listdir(self._track_dir)
        self._track_data = {
            int(file_name[:-4]): np.load(os.path.join(self._track_dir, file_name), allow_pickle=True).item() for file_name in track_file_names
        }

        # Load the file_meta
        self._file_meta = np.load(os.path.join(self._data_dir, 'meta.npy'), allow_pickle=True).item()[self._split]
        self._meta_files = os.listdir(self._meta_dir)

        self._max_cars = max([meta['num_cars'] for meta in self._file_meta.values()])

        # Cache the data if enough memory is available
        self._is_data_cached = kwargs.get('is_data_cached', True)
        self._data_cache = {}

    def __len__(self):
        # return sum([
        #     file_meta['file_num_samples']
        #     for file_id, file_meta in self._file_meta.items()
        # ])
        return len(self._file_ids)

    def _load_file(self, file_id):
        # if self._is_data_cached and file_id in self._data_cache:
        #     data = self._data_cache[file_id]
        # else:
        #     data = np.load(os.path.join(self._sample_dir, f"{file_id}.npy"), allow_pickle=True).item()
        #     self._data_cache[file_id] = data
        # return data
        # Load pkl file
        with open(os.path.join(self._meta_dir, f"{file_id}.pkl"), "rb") as fp:
            meta_file_data = pickle.load(fp)
        data = np.load(os.path.join(self._sample_dir, f"{file_id}.npy"), allow_pickle=True).item()

        return data, meta_file_data

    def __getitem__(self, idx):
        try:
            # idx = 0 #todo
            # Load the data
            file_id = self._file_ids[idx // 100]
            data, meta_file_data = self._load_file(file_id)
            file_sample_meta, probability_map = meta_file_data.values()
            probability_map /= probability_map.sum()
            track = self._track_data[data['track_id']]
            border_points = np.concatenate([
                track['left_track'], track['right_track']
            ], axis=0)
            racing_line_points = track['racing_line']

            # Sample a sequence
            # try:
            #     frame_idx = np.random.choice(len(meta_file_data), p=probability_map)
            # except Exception as e:
            #     print(e)
            #     frame_idx = 0
            # # frame_idx = 0 # todo
            # random frame_idx
            frame_idx = np.random.randint(0, len(meta_file_data))
            left_bound, right_bound, focus_car, car_filter, border_points_mask, racing_line_points_mask  = file_sample_meta[frame_idx]
            right_bound = min(self._future_frames, right_bound - left_bound) + left_bound

            sample_data = data['data'][left_bound:right_bound, car_filter]
            racing_line_points = racing_line_points[racing_line_points_mask]
            border_points = border_points[border_points_mask]

            # Detect large jumps in position # TODO move this to preprocessing...
            sample_data_diff = np.abs(sample_data[1:, :, 0] - sample_data[:-1, :, 0]) > 10
            assert not sample_data_diff.any(), "Large jumps detected in the data"

            # Here we lose the notion of the "focus car".
            # Instead we will take the mean of the racing line points.
            mean_point = border_points.mean(0)

            border_points = border_points - mean_point[None, :]
            racing_line_points = racing_line_points - mean_point[None, :]
            sample_data = sample_data - mean_point[None, None, None, :]

            # Construct 3d rotation matrix
            # angle = np.arctan2(focus_car_direction[0], focus_car_direction[2])
            # Random angle
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

            # Scale the data
            scale = 100
            sample_data /= scale
            border_points /= scale
            racing_line_points /= scale

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
                'start_skip_frames': 2,
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
        except Exception as e:
            # print("Error with index", idx)#
            # Get random idx instead < len
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)


if __name__ == '__main__':
    num_frames=50
    # Test the ACCDataset
    dataset = ACCDataset(num_frames=num_frames, split='train')
    # Create a dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    max_count = 100
    for i, x in enumerate(loader):
        # if x['sample_data'].shape[2] <= 1:
        #     continue
        # break
        if i >= max_count:
            break
