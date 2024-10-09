import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def build_track_data(data, width=15, num_track_points=1000, num_racing_line_points=1000):
    # The goal is to build the track points, and the expected racing line.
    # Assuming data is a list of races, where each race has a list of car data.
    # The car data is a numpy array.    assert num_cars is not None
    left_border_points = None
    right_border_points = None
    center_points = None
    for race_data in tqdm(data, position=1, leave=False):
        data_samples = race_data['data'].transpose(1, 0, 2, 3)
        for car_data in tqdm(data_samples, position=2, leave=False):
            if car_data.shape[0] == 0:
                continue
            coords = car_data[:, 0]
            # Mask out the points where the wheels are the exact same position...
            mask = (car_data[:, 1] == car_data[:, 2]) & (car_data[:, 2] == car_data[:, 3]) & (car_data[:, 3] == car_data[:, 4])
            mask = mask.any(1)
            coords = coords[~mask, :]
            # Remove duplicate points (otherwise splprep fails)
            # coords = np.unique(coords, axis=0)
            # Change the y-axis to z-axis
            coords = np.concatenate((coords[:, 0:1], coords[:, 2:3], coords[:, 1:2]), axis=1)

            # Get the direction vectors by comparing sequential points
            direction_vectors = coords[5:, [0, 1]] - coords[0:-5, [0, 1]]
            # Get the normals
            normals = np.array([-direction_vectors[:, 1], direction_vectors[:, 0]]).T

            np.seterr(divide='ignore', invalid='ignore')
            # Convert to unit normals
            normals = normals / np.linalg.norm(normals, axis=1)[:, None]
            np.seterr(divide='warn', invalid='warn')

            # Normals are of shape (n, 2) and coords are of shape (n, 3)
            normals = np.concatenate((normals, np.zeros((len(normals), 1))), axis=1)

            # Combine numpy arrays
            local_left_border_points = coords[5:] - width / 2 * normals
            local_right_border_points = coords[5:] + width / 2 * normals
            local_center_points = coords[5:]

            # Remove nan values
            nan_indices = np.isnan(local_left_border_points).any(axis=1)
            local_left_border_points = local_left_border_points[~nan_indices]
            local_right_border_points = local_right_border_points[~nan_indices]
            local_center_points = local_center_points[~nan_indices]

            # Concatenate the local border points
            left_border_points = local_left_border_points if left_border_points is None else np.concatenate(
                (left_border_points, local_left_border_points), axis=0)
            right_border_points = local_right_border_points if right_border_points is None else np.concatenate(
                (right_border_points, local_right_border_points), axis=0)
            center_points = local_center_points if center_points is None else np.concatenate(
                (center_points, local_center_points), axis=0)

    # Clean up the left_border_points and right_border_points
    left_border_points = np.unique(left_border_points, axis=0)
    right_border_points = np.unique(right_border_points, axis=0)
    center_points = np.unique(center_points, axis=0)

    assert center_points.shape[0] > 0, "No points."

    # Remove nan values
    left_border_points = left_border_points[~np.isnan(left_border_points).any(axis=1)]
    right_border_points = right_border_points[~np.isnan(right_border_points).any(axis=1)]
    center_points = center_points[~np.isnan(center_points).any(axis=1)]

    # KMeans cluster the left and right border points to num_points // 2
    # Select a point from the cluster, not the center
    k_means_left = KMeans(n_clusters=num_track_points // 2).fit(left_border_points)
    left_border_points = np.array(
        [left_border_points[k_means_left.labels_ == i][0] for i in range(num_track_points // 2)])

    k_means_right = KMeans(n_clusters=num_track_points // 2).fit(right_border_points)
    right_border_points = np.array(
        [right_border_points[k_means_right.labels_ == i][0] for i in range(num_track_points // 2)])

    k_means_center = KMeans(n_clusters=num_racing_line_points).fit(center_points)
    center_points = np.array(
        [center_points[k_means_center.labels_ == i][0] for i in range(num_racing_line_points)])

    return {
        "left_track": np.concatenate((left_border_points[:, 0:1], left_border_points[:, 2:3], left_border_points[:, 1:2]), axis=1),
        "right_track": np.concatenate((right_border_points[:, 0:1], right_border_points[:, 2:3], right_border_points[:, 1:2]), axis=1),
        "racing_line": np.concatenate((center_points[:, 0:1], center_points[:, 2:3], center_points[:, 1:2]), axis=1)
    }


if __name__ == '__main__':
    # data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/'
    # save_track_dir = 'builder/track_data/'
    data_dir = "datasets/builder/ghost_data/"
    save_track_dir = 'datasets/builder/track_data/'
    os.makedirs(save_track_dir, exist_ok=True)
    file_names = os.listdir(data_dir)
    # If track_ids.npy file exists
    if os.path.exists('datasets/builder/track_ids.npy'):
        track_ids = np.load('datasets/builder/track_ids.npy', allow_pickle=True).item()
    else:
        track_ids = {}
        for file_name in tqdm(file_names):
            track_id = np.load(f"{data_dir}{file_name}", allow_pickle=True).item()['track_id']
            track_ids[track_id] = track_ids.get(track_id, []) + [file_name]
        # Save track_ids
        np.save('datasets/builder/track_ids.npy', track_ids, allow_pickle=True)

    for track_id in tqdm(list(track_ids.keys()), position=0, leave=False):
        try:
            response = build_track_data([
                np.load(f"{data_dir}{track_ids[track_id][i]}", allow_pickle=True).item()
                for i in range(min(len(track_ids[track_id]), 10))
            ])
            # Save to track_dir
            np.save(f"{save_track_dir}{track_id}.npy", response, allow_pickle=True)
        except Exception as e:
            print(f"Error: {e}")
            continue
