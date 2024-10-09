import pickle
import os

# If does not exist create pkl folder
if not os.path.exists("json"):
    os.mkdir("json")

data_name = "3_player_sm"
data = pickle.load(open(f"pkl/{data_name}.pkl", "rb"))

# Plot the list data items
import matplotlib.pyplot as plt
for outer_key in ["physics", "graphics"]:
    for inner_key in data["list"][0][outer_key].__dict__.keys():
        # Check if key is numeric
        try:
            data["list"][0][outer_key].__dict__[inner_key] + 1
        except:
            continue

        plt.plot([item[outer_key].__dict__[inner_key] for item in data["list"]])
        plt.title(f"{outer_key} {inner_key}")
        plt.show()

data = {
    "static": str(data["static"]),
    "list": [{
        "physics": str(item["physics"]),
        "graphics": str(item["graphics"])
    } for item in data["list"]]
}

#
# # Save to JSON
# import json
#
# with open(f"json/{data_name}.json", "w") as f:
#     json.dump(data, f)
#     print(f"json/{data_name}.json")
