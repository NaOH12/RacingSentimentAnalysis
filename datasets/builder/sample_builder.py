import os
import pickle
from random import sample

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/session_data/'
    track_dir = 'C:\\Users\\noahl\Documents\ACCDataset/track_data/'
    meta_dir = 'C:\\Users\\noahl\Documents\ACCDataset/meta_data/'

    # Store dataset related meta e.g. number of nearby cars
    # which can be used as a metric for biasing the sampling.
    num_frames = 200
    max_distance = 50
    max_square_distance = max_distance**2

    for split in tqdm(['train', 'val', 'test'], desc='Split', leave=False, position=0):
        file_meta = {}
        for file_name in tqdm(os.listdir(f"{data_dir}{split}/"), desc='File', leave=False, position=1):
            # If file already exists
            if os.path.exists(f"{meta_dir}{split}/{file_name[:-4]}.pkl"):
                # Load pkl
                with open(f"{meta_dir}{split}/{file_name[:-4]}.pkl", "rb") as fp:
                    samples = pickle.load(fp)
                    try:
                        samples = samples['samples']
                    except Exception as e:
                        continue

                # if isinstance(samples, dict):
                #     continue
                #
                # if len(samples) == 0:
                #     continue
                #
                # if len(samples[0]) == 7:
                #     # Re-normalize the probability map
                #     probabilities = []
                #     for i, sample in enumerate(samples):
                #         samples[i] = list(sample)
                #         probabilities.append(samples[i].pop(3))
                #         samples[i] = tuple(samples[i])
                #     probability_map = np.array(probabilities)
                #     probability_map /= probability_map.sum()
                #
                #     save_dict = {
                #         'samples': samples,
                #         'probability_map': probability_map
                #     }
                #
                #     # Re-save the samples
                #     with open(f"{meta_dir}{split}/{file_name[:-4]}.pkl", "wb") as fp:
                #         pickle.dump(save_dict, fp)

                file_id = file_name[:-4]
                data = np.load(f"{data_dir}{split}/{file_name}", allow_pickle=True).item()
                sample_data = data['data'].astype(np.float32)
                file_meta[file_id] = {
                    'num_samples': len(samples),
                    'num_cars': sample_data.shape[1],
                    'track_id': data['track_id']
                }
            else:
                file_id = file_name[:-4]
                data = np.load(f"{data_dir}{split}/{file_name}", allow_pickle=True).item()
                sample_data = data['data'].astype(np.float32)
                # invalids = data['invalids'].astype(np.bool)
                invalids = np.ones((sample_data.shape[0], sample_data.shape[1]), dtype=np.bool)
                invalids[1:] = ((sample_data[1:, :, 0, :].round(3) - sample_data[:-1, :, 0, :].round(3)) == 0).all(-1)

                # Track data
                track_id = data['track_id']
                # Load track data
                track_data = np.load(f"{track_dir}{track_id}.npy", allow_pickle=True).item()
                border_points = np.concatenate(
                    [track_data['left_track'], track_data['right_track']], axis=0
                ).astype(np.float32)
                racing_line_points = track_data['racing_line'].astype(np.float32)

                # Construct probability map for the data based on car proximity and invalids.
                # Invalid regions should not be considered for sampling.
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
                except Exception as e:
                    pass

                if (~invalids).any() == False:
                    break

                # Iterate over the probability map and record consecutive probable frames.
                samples = []
                probabilities = []
                for car_id in tqdm(range(sample_data.shape[1]), desc='Car', leave=False, position=2):
                    left_bound = None
                    for i in tqdm(range(0, sample_data.shape[0]), desc='Frame', leave=False, position=3):
                        if left_bound is None and probability_map[i, car_id] != 0:
                            left_bound = i
                        elif left_bound is not None and (i - left_bound) == num_frames:
                            # Get the combined probability over the segment
                            segment_prob = probability_map[left_bound:i, car_id].sum()
                            # Get the nearby car ids
                            car_filter = (
                                car_square_distances[left_bound:left_bound+num_frames, car_id].min(0)
                            ) < max_square_distance
                            # Get the track point ids

                            # Filter data by computing square distances across the sequence, and selecting
                            # data that exists within the min distance at least once in the episode.
                            # Filter the track points
                            border_points_sqr_distances = ((
                                border_points[:, None] - sample_data[left_bound:left_bound+num_frames, car_id, 0][None, ...]) ** 2
                            ).sum(-1)
                            border_points_sqr_distances = border_points_sqr_distances.min(-1)
                            border_points_mask = border_points_sqr_distances < max_square_distance
                            racing_line_sqr_distances = ((
                                racing_line_points[:, None] - sample_data[left_bound:left_bound+num_frames, car_id, 0][None, ...]) ** 2
                            ).sum(-1)
                            racing_line_sqr_distances = racing_line_sqr_distances.min(-1)
                            racing_line_points_mask = racing_line_sqr_distances < max_square_distance

                            samples.append((left_bound, i, car_id, car_filter, border_points_mask, racing_line_points_mask))
                            probabilities.append(segment_prob)
                            left_bound = i
                        elif left_bound is not None and probability_map[i, car_id] == 0:
                            left_bound = None

                if len(samples) == 0:
                    continue

                # Re-normalize the probability map
                probability_map = np.array(probabilities)
                probability_map /= probability_map.sum()

                save_dict = {
                    'samples': samples,
                    'probability_map': probability_map
                }

                # Save the samples
                with open(f"{meta_dir}{split}/{file_id}.pkl", "wb") as fp:
                    pickle.dump(save_dict, fp)

                file_meta[file_id] = {
                    'num_samples': len(samples),
                    'num_cars': sample_data.shape[1],
                    'track_id': data['track_id']
                }

        total_samples = sum([meta['num_samples'] for meta in file_meta.values()])
        for file_id, meta in file_meta.items():
            meta['file_prob'] = meta['num_samples'] / total_samples

        # Save the file meta
        np.save(f'C:\\Users\\noahl\Documents\ACCDataset/meta_{split}.npy', file_meta, allow_pickle=True)

    # Load the file_meta files
    train_meta = np.load('C:\\Users\\noahl\Documents\ACCDataset/meta_train.npy', allow_pickle=True).item()
    val_meta = np.load('C:\\Users\\noahl\Documents\ACCDataset/meta_val.npy', allow_pickle=True).item()
    test_meta = np.load('C:\\Users\\noahl\Documents\ACCDataset/meta_test.npy', allow_pickle=True).item()

    meta_dict = {
        'train': train_meta,
        'val': val_meta,
        'test': test_meta
    }

    # Save the meta_dict
    np.save('C:\\Users\\noahl\Documents\ACCDataset/meta_dict.npy', meta_dict, allow_pickle=True)
