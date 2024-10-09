import os
from random import sample

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/race_data/'
    track_dir = 'C:\\Users\\noahl\Documents\ACCDataset/track_data/'
    meta_dir = 'C:\\Users\\noahl\Documents\ACCDataset/meta_data/'
    os.makedirs(meta_dir, exist_ok=True)

    # Store dataset related meta e.g. number of nearby cars
    # which can be used as a metric for biasing the sampling.
    file_meta = {}
    prior_weight = 1

    for file_name in tqdm(os.listdir(data_dir)):
        file_id = file_name[:-4]
        data = np.load(f"{data_dir}{file_name}", allow_pickle=True).item()

        sample_data = data['data']
        invalids = data['invalids']

        coords = sample_data[:, :, 0]
        n, num_cars, _ = coords.shape

        # distances = np.zeros((n, num_cars, num_cars))
        # Compute the distances for each car to each other car
        distances = np.linalg.norm(coords[:, :, None, ...] - coords[:, None, ...], axis=-1)
        distances[invalids] = np.inf

        proximity_cars = (distances < 10).sum(axis=-1) - 1
        proximity_cars[invalids] = 0

        # indices = (~invalids).nonzero()
        # Shape is (n, num_cars)
        weight = (distances < 20).sum(axis=-1) - 1 + prior_weight

        # # Save the file meta
        # np.save(f'{meta_dir}{file_id}.npy', {
        #     'indices': indices,
        #     'probabilities': weight / weight.sum(),
        # }, allow_pickle=True)

        file_meta[file_id] = {
            'file_weight': weight.sum(),
            'file_num_samples': (~invalids).shape[0],
            'file_num_cars': num_cars,
        }

    total_weight = sum([meta['file_weight'] for meta in file_meta.values()])
    for file_id, meta in file_meta.items():
        meta['file_prob'] = meta['file_weight'] / total_weight

    # Save the file meta
    np.save('C:\\Users\\noahl\Documents\ACCDataset/file_meta.npy', file_meta, allow_pickle=True)
