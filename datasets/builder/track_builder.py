import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def build_track_data(data, num_track_points=1000):
    out_dict = {}
    for (prefix, coords), out_name in zip(data.items(), ['left_track', 'right_track', 'racing_line']):
        k_means = KMeans(n_clusters=num_track_points).fit(coords)
        coords = np.array([
            coords[k_means.labels_ == i][0]
            for i in range(num_track_points)
        ])
        coords = np.unique(coords, axis=0)
        out_dict[out_name] = coords.astype(np.float32)

    return out_dict


if __name__ == '__main__':
    # data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/'
    # save_track_dir = 'builder/track_data/'
    data_dir = "datasets/builder/track_data/"
    save_track_dir = 'C:\\Users\\noahl\Documents\ACCDataset\\track_data/'
    os.makedirs(save_track_dir, exist_ok=True)
    file_names = os.listdir(data_dir)
    # If track_ids.npy file exists
    track_ids = {}
    for file_name in tqdm(file_names):
        track_id = np.load(f"{data_dir}{file_name}", allow_pickle=True).item()['track_id']
        track_ids[track_id] = track_ids.get(track_id, []) + [file_name]

    track_id = 12
    assert track_id in track_ids, f"Track {track_id} not found."

    # Get the left, right and ghost data
    data_file_names = {}
    for prefix in ['left', 'right', 'ghost']:
        file_name = [
            file_name for file_name in track_ids[track_id] if file_name.split('_')[0] == prefix
        ][0]
        data_file_names[prefix] = file_name

    data = {
        prefix: np.load(f"{data_dir}{data_file_names[prefix]}", allow_pickle=True).item()['data']
        for prefix in ['left', 'right', 'ghost']
    }

    # Visualize the data
    import matplotlib.pyplot as plt
    # Increase res
    plt.figure(figsize=(10, 10))
    vis_prefixes = ['left', 'right', 'ghost']
    range_data = {
        'left': [200, -1],
        'right': [200, -1],
        'ghost': [300, 1300]
    }
    for prefix, colour in zip(['left', 'right', 'ghost'], ['r', 'b', 'g']):
        if prefix in vis_prefixes:
            if prefix == 'ghost':
                car_idx=0
                plt.scatter(
                    data[prefix][range_data[prefix][0]:range_data[prefix][1], car_idx, 0, 0],
                    data[prefix][range_data[prefix][0]:range_data[prefix][1], car_idx, 0, 2],
                    label=prefix, color=colour, s=0.1
                )
                data[prefix] = data[prefix][range_data[prefix][0]:range_data[prefix][1], car_idx, 0]
            else:
                plt.scatter(
                    data[prefix][range_data[prefix][0]:range_data[prefix][1], 0, 0, 0],
                    data[prefix][range_data[prefix][0]:range_data[prefix][1], 0, 0, 2],
                    label=prefix, color=colour, s=0.1
                )
                data[prefix] = data[prefix][range_data[prefix][0]:range_data[prefix][1], 0, 0]

    plt.legend()
    plt.show()

    # response = build_track_data(data)
    # np.save(f"{save_track_dir}{track_id}.npy", response, allow_pickle=True)
