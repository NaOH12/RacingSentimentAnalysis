from pyaccsharedmemory import accSharedMemory
import os

asm = accSharedMemory()
sm = asm.read_shared_memory()

# If does not exist create pkl folder
if not os.path.exists("pkl"):
    os.mkdir("pkl")

data = {
    "static": sm.Static,
    "list": []
}

try:
    while True:
        if sm is not None:
            # print("Physics:")
            # print(f"Pad life: {sm.Physics.pad_life}")
            #
            # print("Graphics:")
            # print(f"Strategy tyre set: {sm.Graphics.penalty.name}")
            #
            # print("Static: ")
            # print(f"Max RPM: {sm.Static.max_rpm}")
            data["list"].append({"physics": sm.Physics, "graphics": sm.Graphics})

        sm = asm.read_shared_memory()
# Catch user ctrl c
except KeyboardInterrupt:
    print("KeyboardInterrupt")
    pass

# Save dictionary as pickle with time in name
import pickle
import time

pickle_file = f"pkl/shared_memory_{time.time()}.pkl"
with open(pickle_file, "wb") as f:
    pickle.dump(data, f)
    print(f"Data saved to {pickle_file}")

asm.close()
