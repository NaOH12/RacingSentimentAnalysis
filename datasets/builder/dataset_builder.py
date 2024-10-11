import os
import pickle

import numpy as np
from tqdm import tqdm

from rpy_parser import RaceData

# rpy_dir = 'rpy/'
rpy_dir = 'D:\Libraries\Documents\Freiburg Informatiks Work\ACCBCDataset\\replays/'
# rpy_dir = 'C:\\Users\\noahl\Documents\Assetto Corsa Competizione\Replay\Saved/'
# data_dir = 'data/'
# data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/'
# rpy_dir = 'datasets/builder/rpy/ghost_replays/'
# rpy_dir = 'datasets/builder/rpy/border_replays/'
# data_dir = 'datasets/builder/track_data/'
data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/race_data/'
os.makedirs(data_dir, exist_ok=True)

# Get files in data/
files = os.listdir(data_dir)
# Remove .npy extension
saved_files = [file[:-4] for file in files]

# For each file in rpy directory
for file in tqdm(os.listdir(rpy_dir)):
    file_idx = file[:-4]
    if file_idx in saved_files:
        continue
    file_name = rpy_dir + file
    for skip_start, is_extended in [(False, False), (False, True), (True, True), (True, False)]:
        try:
            race_data = RaceData(file_name, is_extended=is_extended)
            data = race_data.build_coord_data()
            if np.prod(data['data'].shape) == 0:
                with open('error_log.txt', 'a') as f:
                    f.write(f'Error in file {file_name}: No data\n')
                continue
            data['track_id'] = race_data.session_info['track_id']
            # Save data to file with numpy
            np.save(f"{data_dir}{file[:-4]}.npy", data, allow_pickle=True)
        except Exception as e:
            print(f'Error in file {file_name}: {e}')
            # Write/append to error log file
            with open('error_log.txt', 'a') as f:
                f.write(f'Error in file {file_name}: {e}\n')
            continue
