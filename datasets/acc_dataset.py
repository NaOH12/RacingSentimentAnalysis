import os

import numpy as np
from sympy.stats import sample
from sympy.stats.rv import probability
from torch.utils.data import Dataset


class ACCDataset(Dataset):
    def __init__(self, *args, **kwargs):
        # Number of future frames to predict
        self._future_frames = kwargs.get('future_frames', 20)
        # Skip the first few frames to get initial direction info
        self._start_skip_frames = kwargs.get('start_skip_frames', 2)
        # Limit the number of racing line points and cars to consider
        self._max_distance = kwargs.get('max_distance', 50)
        self._max_square_distance = self._max_distance ** 2

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

    def __len__(self):
        return sum([
            file_meta['file_num_samples'] - self._future_frames - self._start_skip_frames
            for file_id, file_meta in self._file_meta.items()
        ])

    def __getitem__(self, idx):
        # Here idx is enough to select the data file,
        # but we wish to further sample a start frame within the file later.

        # Get the file_id from the idx
        count = 0
        for file_id, file_meta in self._file_meta.items():
            count+= file_meta['file_num_samples'] - self._future_frames - self._start_skip_frames
            if idx < count:
                idx = count - idx
                break

        # Load the file
        data = np.load(os.path.join(self._sample_dir, f"{file_id}.npy"), allow_pickle=True).item()
        sample_data = data['data']
        invalids = data['invalids']
        # Track data
        track_id = data['track_id']
        track_data = self._track_data[track_id]
        left_track = track_data['left_track']
        right_track = track_data['right_track']
        racing_line = track_data['racing_line']

        # Construct probability map for the data based on car proximity.
        # Compute square distance between cars
        car_square_distances = ((
            sample_data[self._start_skip_frames:-self._future_frames, None, :, 0] -
            sample_data[self._start_skip_frames:-self._future_frames, :, None, 0]
        ) ** 2).sum(-1)
        probability_map = (1 - (car_square_distances / car_square_distances.max())).sum(-1)
        probability_map /= probability_map.sum()
        # Given the 2d probability map, sample both dimensions
        frame_idx, focus_car_idx = np.unravel_index(
            np.random.choice(probability_map.size, p=probability_map.ravel()), probability_map.shape
        )
        frame_idx += self._start_skip_frames

        # Filter data by computing square distances across the sequence, and selecting
        # data that exists within the min distance at least once in the episode.
        racing_line_sqr_distances = ((racing_line[:, None] - sample_data[frame_idx:frame_idx+self._future_frames, focus_car_idx, 0][None, ...]) ** 2).sum(-1)
        racing_line_sqr_distances = racing_line_sqr_distances.min(-1)
        racing_line = racing_line[racing_line_sqr_distances < self._max_square_distance]
        car_filter = (
            car_square_distances[frame_idx:frame_idx+self._future_frames, focus_car_idx].min(0)
        ) < self._max_square_distance
        # Filter the data
        sample_data = sample_data[:, car_filter]
        invalids = invalids[:, car_filter]
        # Get new focus car index
        focus_car_idx = np.where(np.where(car_filter)[0] == focus_car_idx)[0][0]

        # Select the previous frames for direction calculation
        prev_frame = sample_data[frame_idx-self._start_skip_frames]
        prev_invalids = invalids[frame_idx-self._start_skip_frames]

        # Select the sequence data
        sample_data = sample_data[frame_idx:frame_idx+self._future_frames]
        invalids = invalids[frame_idx:frame_idx+self._future_frames]

        # TODO Assert invalids

        # Center translation
        translation_vector = sample_data[0, focus_car_idx, 0]
        racing_line = racing_line - translation_vector[None, :]
        prev_frame = prev_frame - translation_vector[None, None, :]
        sample_data = sample_data - translation_vector[None, None, None, :]

        # Rotate the data based on focus car direction. Shape is (3,)
        # Focus car orientation
        focus_car_direction = sample_data[0, focus_car_idx, 1] - sample_data[0, focus_car_idx, 3]
        # Normalize
        focus_car_direction /= np.linalg.norm(focus_car_direction)

        # Construct 3d rotation matrix
        angle = np.arctan2(focus_car_direction[0], focus_car_direction[2])
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        # Sample data shape (n, n_cars, c, 3) -> (n * n_cars * c, 3)
        sample_data_shape = sample_data.shape
        sample_data = sample_data.reshape(-1, 3)

        # Rotate the data
        sample_data = np.matmul(sample_data, rotation_matrix)
        racing_line = np.matmul(racing_line, rotation_matrix)
        focus_car_direction = np.matmul(focus_car_direction, rotation_matrix)

        sample_data = sample_data.reshape(*sample_data_shape)

        # # Visualize
        # import matplotlib.pyplot as plt
        # vis_data = sample_data[0:1, :, 0].reshape(-1, sample_data.shape[3])
        # # Plot the racing line in green
        # plt.scatter(racing_line[:, 0], racing_line[:, 2], c='g')
        # plt.scatter(vis_data[:, 0], vis_data[:, 2])
        # # Visualize direction
        # plt.arrow(
        #     sample_data[0, focus_car_idx, 0, 0], sample_data[0, focus_car_idx, 0, 2],
        #     focus_car_direction[0], focus_car_direction[2],
        #     color='r', head_width=2, head_length=20
        # )
        # plt.show()
        # plt.close()

        # Construct the training data
        return {
            # The directions of the cars with magnitude
            'directions': sample_data[0, focus_car_idx, 1] - sample_data[0, focus_car_idx, 3],
            # Just the first samples
            'car_positions': sample_data[0:1],
            # The "optimal" racing line to provide some guidance.
            'racing_line': racing_line,
        }, {
            'car_positions': sample_data[1:], # The rest of the samples
        }


if __name__ == '__main__':
    # Test the ACCDataset
    dataset = ACCDataset()

    for i in range(len(dataset)):
        response = dataset[i]
        break