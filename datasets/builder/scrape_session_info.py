import os

import numpy as np

# Get the files in the data directory
data_dir = 'data/'
files = os.listdir(data_dir)
# Remove .npy extension
file_ids = [file[:-4] for file in files]

session_dict = {}
# If session dict file exists, load it
if 'session_dict.npy' in files:
    session_dict = np.load(data_dir + 'session_dict.npy', allow_pickle=True).item()

# For each file in the data directory
for file_id in file_ids:
    url = f"www.accreplay.com/replays/{file_id}"
    #
