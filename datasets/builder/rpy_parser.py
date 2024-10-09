import re
import struct
import matplotlib.pyplot as plt
import numpy
import numpy as np


class ByteCounter:
    def __init__(self, end_pos):
        self._count = 0
        self._end_pos = end_pos

    def increment(self, num_bytes):
        self._count += num_bytes
        # If at end + 1 then set to start
        if self._count == self._end_pos:
            self._count = 0

    # Count as a property
    @property
    def count(self):
        return self._count

    def assert_start(self):
        assert self._count == 0, f"Byte counter is not at the start of the file"


class Packet:
    def __init__(self, file, is_extended=False):
        self.is_extended = is_extended
        # Read the data (bytes)
        self._data = self._read_data(file)
        # Create a byte counter
        self._byte_counter = ByteCounter(len(self._data))

    def _read_data(self, file):
        raise NotImplementedError

    def _cast_vector3f(self):
        start = self._byte_counter.count
        # Increment the byte counter
        self._byte_counter.increment(12)
        return [
            struct.unpack('<f', self._data[start:start + 4])[0],
            struct.unpack('<f', self._data[start + 4:start + 8])[0],
            struct.unpack('<f', self._data[start + 8:start + 12])[0]
        ]

    def _cast_string(self):
        start = self._byte_counter.count
        length = struct.unpack('<I', self._data[start:start + 4])[0]
        # Assert the length is not beyond the byte data
        assert length <= len(self._data) - 4, f"String length {length} is way too long"
        # Increment the byte counter
        self._byte_counter.increment(4 + length)
        # Return the string and end pos
        return self._data[start + 4:start + 4 + length].decode("utf-8")

    def _cast_unknown_arr(self, type_id):
        start = self._byte_counter.count
        num_items = struct.unpack('<I', self._data[start:start + 4])[0]
        # Increment the byte counter
        self._byte_counter.increment(4 + num_items * 4)
        return [
            struct.unpack(type_id, self._data[i:i + 4])[0]
            for i in range(start + 4, start + 4 + num_items * 4)
        ]

    def _cast_byte(self):
        start = self._byte_counter.count
        self._byte_counter.increment(1)
        return struct.unpack('<B', self._data[start:start + 1])[0]

    def _cast_ushort(self):
        start = self._byte_counter.count
        self._byte_counter.increment(2)
        return struct.unpack('<H', self._data[start:start + 2])[0]

    def _cast_short(self):
        start = self._byte_counter.count
        self._byte_counter.increment(2)
        return struct.unpack('<h', self._data[start:start + 2])[0]

    def _cast_double(self):
        start = self._byte_counter.count
        self._byte_counter.increment(8)
        return struct.unpack('<d', self._data[start:start + 8])[0]

    def _cast_fixed_arr(self, num_items, type_id):
        # Num items here refers to number of 4-byte items
        if type_id in ['<h', '<H']:
            num_items = num_items * 2  # 2 bytes per item (short)
            num_bytes = 2
        elif type_id in ['<B', '<b']:
            num_items = num_items * 4
            num_bytes = 1
        elif type_id in ['<d']:
            num_bytes = 8
        else:
            num_bytes = 4
        start = self._byte_counter.count
        self._byte_counter.increment(num_items * num_bytes)
        return np.array([
            struct.unpack(
                type_id, self._data[start + i * num_bytes:start + (i + 1) * num_bytes]
            )[0] for i in range(num_items)
        ])


class Packet104(Packet):

    def _read_data(self, f):
        data_size = 3 * 4  # Vector3f
        data_size += 8 * 4 if self.is_extended else 2 * 4
        data_size += 4 * 3 * 4  # 4 * Vector3f
        data_size += 9 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            "coords": self._cast_vector3f()
        }
        if self.is_extended:
            # self._cast_byte()
            # self._cast_byte()
            cast_data = cast_data | {
                "arr1_unknown_1": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_2": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_3": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_4": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_5": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_6": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_7": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_8": self._cast_fixed_arr(1, unknown_type),
                # "arr1_unknown_8": self._cast_fixed_arr(1, "<h")
            }
            # self._cast_byte()
            # self._cast_byte()
            # self._cast_byte()
        else:
            cast_data = cast_data | {
                "arr1_unknown_1": self._cast_fixed_arr(1, unknown_type),
                "arr1_unknown_2": self._cast_fixed_arr(1, unknown_type)
            }
        cast_data = {
            **cast_data,
            "contactPoint1": self._cast_vector3f(),
            "contactPoint2": self._cast_vector3f(),
            "contactPoint3": self._cast_vector3f(),
            "contactPoint4": self._cast_vector3f(),
            "unknown_1": self._cast_fixed_arr(1, unknown_type),
            "maybe_steer_angle": self._cast_fixed_arr(1, "<f"),
            # "arr2": self._cast_fixed_arr(7, "<f")
            "unknown_2": self._cast_fixed_arr(1, unknown_type),
            "unknown_3": self._cast_fixed_arr(1, unknown_type),
            "unknown_4": self._cast_fixed_arr(1, unknown_type),
            "unknown_5": self._cast_fixed_arr(1, unknown_type),
            "unknown_6": self._cast_fixed_arr(1, unknown_type),
            # "unknown_7": self._cast_fixed_arr(1, unknown_type),
            # "104_clock": self._cast_fixed_arr(1, unknown_type),
            "104_clock": self._cast_fixed_arr(1, "<d"),
        }
        self._byte_counter.assert_start()
        return cast_data


class Packet48(Packet):
    def _read_data(self, f):
        data_size = 12 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            "steer_angle": self._cast_fixed_arr(1, "<f")[0],
            "brake": self._cast_fixed_arr(1, unknown_type),
            "48_unknown": self._cast_fixed_arr(1, unknown_type),
            "speed": self._cast_fixed_arr(1, unknown_type),
            # "arr3": self._cast_fixed_arr(8, "<f")
            "48_unknown_1": self._cast_fixed_arr(1, unknown_type),
            "48_unknown_2": self._cast_fixed_arr(1, unknown_type),
            "48_unknown_3": self._cast_fixed_arr(1, unknown_type),
            "48_unknown_4": self._cast_fixed_arr(1, unknown_type),
            "48_unknown_5": self._cast_fixed_arr(1, unknown_type),
            "48_unknown_6": self._cast_fixed_arr(1, unknown_type),
            # "48_unknown_7": self._cast_fixed_arr(1, unknown_type),
            # "48_clock": self._cast_fixed_arr(1, unknown_type),
            "48_clock": self._cast_fixed_arr(1, "<d"),
            # "arr3": self._cast_fixed_arr(12, all_type)
        }
        self._byte_counter.assert_start()
        return cast_data


class Packet72(Packet):
    def _read_data(self, f):
        data_size = 18 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            "72_unknown_1": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_2": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_3": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_4": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_5": self._cast_fixed_arr(1, unknown_type),
            "gas": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_7": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_8": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_9": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_10": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_11": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_12": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_13": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_14": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_15": self._cast_fixed_arr(1, unknown_type),
            "72_unknown_16": self._cast_fixed_arr(1, unknown_type),
            "72_clock": self._cast_fixed_arr(1, "<d"),
        }
        self._byte_counter.assert_start()
        return cast_data


class Packet24(Packet):
    def _read_data(self, f):
        data_size = 6 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            # "arr5": self._cast_fixed_arr(6, '<i')
            "24_unknown_1": self._cast_fixed_arr(1, unknown_type),
            "24_unknown_2": self._cast_fixed_arr(1, unknown_type),
            "24_unknown_3": self._cast_fixed_arr(1, unknown_type),
            "24_unknown_4": self._cast_fixed_arr(1, unknown_type),
            # "24_unknown_5": self._cast_fixed_arr(1, unknown_type),
            # "24_clock": self._cast_fixed_arr(1, unknown_type),
            "24_clock": self._cast_fixed_arr(1, "<d"),
        }
        self._byte_counter.assert_start()
        return cast_data


class Packet16(Packet):
    def _read_data(self, f):
        data_size = 4 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            # "arr6": self._cast_fixed_arr(4, '<f')
            "16_unknown_1": self._cast_fixed_arr(1, unknown_type),
            "16_unknown_2": self._cast_fixed_arr(1, unknown_type),
            # "16_unknown_3": self._cast_fixed_arr(1, unknown_type),
            # "16_clock": self._cast_fixed_arr(1, unknown_type),
            "16_clock": self._cast_fixed_arr(1, "<d"),
        }
        self._byte_counter.assert_start()
        return cast_data


class Packet162(Packet):
    def _read_data(self, f):
        data_size = 4 * 4
        return f.read(data_size)

    def cast(self):
        global unknown_type
        cast_data = {
            # "arr6": self._cast_fixed_arr(4, '<f')
            "16_2_unknown_1": self._cast_fixed_arr(1, unknown_type),
            "16_2_unknown_2": self._cast_fixed_arr(1, unknown_type),
            # "16_unknown_3": self._cast_fixed_arr(1, unknown_type),
            # "16_clock": self._cast_fixed_arr(1, unknown_type),
            "16_2_clock": self._cast_fixed_arr(1, "<d"),
        }
        self._byte_counter.assert_start()
        return cast_data


def parse_string(f):
    length = struct.unpack('<I', f.read(4))[0]
    return f.read(length).decode("utf-8")


def parse_unknown_arr(f, type_id):
    num_items = struct.unpack('<I', f.read(4))[0]
    return [struct.unpack(type_id, f.read(4))[0] for _ in range(num_items)]


def parse_team(f, last_car=False):
    team_meta = {
        "full_name": parse_string(f),
        "short_name": parse_string(f),
        "full_name_2": parse_string(f),
        "arr": numpy.array([struct.unpack('<i', f.read(4))[0] for _ in range(13)]),
        "end": struct.unpack('<i', f.read(4))[0] if not last_car else struct.unpack('<H', f.read(2))[0]
    }
    return team_meta


def parse_driver(f):
    driver_meta = {
        "token": struct.unpack('<H', f.read(2))[0],
        "first_name": parse_string(f),
        "unknown_str": parse_string(f),
        "last_name": parse_string(f),
        "nickname": parse_string(f),
        "short_name": parse_string(f),
        "arr": [struct.unpack('<B', f.read(1))[0] for _ in range(41)],
        "uuid": parse_string(f)
    }
    return driver_meta


def parse_session_info(f):
    return {
        "short_1": struct.unpack('<h', f.read(2))[0],
        "track_id": struct.unpack('<I', f.read(4))[0],
        "type_id": struct.unpack('<I', f.read(4))[0],
        "unknown_arr": [struct.unpack('<f', f.read(4))[0] for _ in range(10)],
        "server_name": parse_string(f),
        "short_2": struct.unpack('<h', f.read(2))[0],
        "unknown_arr_2": [struct.unpack('<f', f.read(4))[0] for _ in range(14)],
        "unknown_arr_3": parse_unknown_arr(f, '<I'),
        "unknown_arr_4": parse_unknown_arr(f, '<I'),
        "start_token": struct.unpack('<I', f.read(4))[0],
        "ambient_temp": struct.unpack('<f', f.read(4))[0],
        "unknown_arr_5": [struct.unpack('<f', f.read(4))[0] for _ in range(2)],
        "road_temp": struct.unpack('<f', f.read(4))[0],
        "unknown_arr_6": [struct.unpack('<f', f.read(4))[0] for _ in range(2)],
        "sunlight_factor": struct.unpack('<f', f.read(4))[0],
        "start_token_2": struct.unpack('<I', f.read(4))[0]
    }


def parse_track_id(f):
    _ = struct.unpack('<h', f.read(2))[0]
    return {
        "track_id": struct.unpack('<I', f.read(4))[0]
    }


def merge_dict_dicts(dict_dicts):
    # Merge a list of dictionaries whose values are dictionaries
    # Dictionaries with the same key will have their (value) dictionaries merged.
    merges = 0
    non_merges = 0
    for dict_dict in dict_dicts[1:]:
        for key, value in dict_dict.items():
            if key in dict_dicts[0]:
                dict_dicts[0][key].update(value)
                merges += 1
            else:
                dict_dicts[0][key] = value
                non_merges += 1

    print(f"Merged {merges} dictionaries and added {non_merges} dictionaries")
    return dict_dicts[0]


def deep_merge(data):
    # Merge a list of dictionaries, and if conflicts arise, recursively merge the dictionaries
    merged = {}
    for d in data:
        for k, v in d.items():
            if k in merged:
                if isinstance(v, dict):
                    merged[k] = deep_merge([merged[k], v])
                else:
                    # Cannot merge non-dictionary values
                    raise ValueError(f"Cannot merge non-dictionary values {merged[k]} and {v}")
            else:
                merged[k] = v

    return merged


class CarData:
    # Enum for the packet types
    PACKET_104 = "104"
    PACKET_48 = "48"
    PACKET_72 = "72"
    PACKET_24 = "24"
    PACKET_16 = "16"
    PACKET_16_2 = "16_2"


    def __init__(self):
        self._packet_data_104 = []
        self._packet_data_48 = []
        self._packet_data_72 = []
        self._packet_data_24 = []
        self._packet_data_16 = []
        self._packet_data_16_2 = []

    def _cast_packet_data(self, enum):
        packet_data = getattr(self, f"_packet_data_{enum}")
        packet_data = [packet.cast() for packet in packet_data]
        return packet_data

    def cast(self):
        data = [
            {d["104_clock"][0]: {**d, **{"104_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_104))},
            {d["48_clock"][0]: {**d, **{"48_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_48))},
            {d["72_clock"][0]: {**d, **{"72_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_72))},
            {d["24_clock"][0]: {**d, **{"24_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_24))},
            {d["16_clock"][0]: {**d, **{"16_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_16))},
            {d["16_2_clock"][0]: {**d, **{"16_2_idx": idx}} for idx, d in
             enumerate(self._cast_packet_data(self.PACKET_16_2))}
        ]


        data = merge_dict_dicts(data)

        # Order the dictionary by keys
        data = dict(sorted(data.items()))

        return data

    def cast_coord_info(self):
        data = {
            d["104_clock"][0]: {**d, **{"104_idx": idx}}
            for idx, d in enumerate(self._cast_packet_data(self.PACKET_104))
        }

        data = {
            key: [
                data.get("coords", None),
                data.get("contactPoint1", None),
                data.get("contactPoint2", None),
                data.get("contactPoint3", None),
                data.get("contactPoint4", None)
            ] for key, data in data.items()
        }

        return data

    def visualize_packet_data(self, enum, start_render=0.0, end_render=1.0):
        packet_data = self._cast_packet_data(enum)
        packet_data = {key: [d[key] for d in packet_data] for key in packet_data[0]}

        # Limit the amount of data to render to the beginning percentage start_render and end_render
        packet_data = {
            key: value[int(len(value) * start_render):int(len(value) * end_render)]
            for key, value in packet_data.items()
        }

        for key, value in packet_data.items():
            # Assert the value is a list
            assert isinstance(value, list), f"Value is not a list {value}"
            # If a vector then plot in 3d
            if type(value[0]) == tuple:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter([v[0] for v in value], [v[1] for v in value], [v[2] for v in value])
                plt.title(f"Packet {enum} {key}")
                plt.show()
            # If the value is a list of lists then plot each list
            elif isinstance(value[0], list):
                for key_idx in range(len(value[0])):
                    # Create array of values for the key index
                    key_values = [v[key_idx] for v in value]
                    # Plot
                    plt.plot(key_values)
                    plt.title(f"Packet {enum} {key} {key_idx}")
                    plt.show()
            # If the value is a list of dictionaries then plot each dictionary
            elif isinstance(value[0], dict):
                for key, _ in value[0].items():
                    plt.plot([v[key] for v in value])
                    plt.title(f"Packet {enum} {key}")
                    plt.show()
            # Else attempt to plot the value
            else:
                plt.plot(value)
                plt.title(f"Packet {enum} {key}")
                plt.show()


    def visualize(self, start_render=0.0, end_render=1.0):
        data = self.cast()

        name_list = set()
        # Iterate over the entire data to first get all of the unique names...
        for packet_id, packet_data in data.items():
            name_list = name_list | set(packet_data.keys())

        name_indexed_data = {
            name: np.array([0]) for name in name_list
        }

        for time_idx, merged_packet in data.items():
            for name in name_list:
                data_item = merged_packet.get(name, name_indexed_data[name][-1:])
                if isinstance(data_item, np.ndarray) is False:
                    data_item = np.array([data_item])

                if len(data_item) > 1:
                    # Compute product
                    data_item = np.prod(data_item, keepdims=True)

                name_indexed_data[name] = np.concatenate([name_indexed_data[name], data_item])

        # Get the start idx and end idx
        start_idx = int(len(name_indexed_data[name]) * start_render)
        end_idx = int(len(name_indexed_data[name]) * end_render)

        # Slice the data
        name_indexed_data = {name: data[start_idx:end_idx] for name, data in name_indexed_data.items()}

        # Plot the data
        fig, axs = plt.subplots(len(name_list), figsize=(10, 110))
        for idx, (name, data) in enumerate(name_indexed_data.items()):
            axs[idx].plot(data)
            axs[idx].set_title(name)
        plt.show()
        plt.close()

        return name_indexed_data


class RaceData:
    def __init__(self, file_name, is_extended=False, skip_start=False):
        self._teams_meta = []
        self._drivers_meta = []
        self._cars = []
        self._track_points = None

        with open(file_name, "rb") as f:
            # self.session_info = parse_session_info(f)
            self.session_info = parse_track_id(f)
            # # Assert start token
            # assert self.session_info["start_token"] == 123
            # assert self.session_info["start_token_2"] == 123

            # # Read the number of cars
            # num_teams = struct.unpack('<I', f.read(4))[0]
            # print(f"Found {num_teams} teams")
            # for team_id in range(num_teams):
            #     self._teams_meta.append(parse_team(f, last_car=team_id == num_teams - 1))
            #
            # # Assert start token
            # assert struct.unpack('<I', f.read(4))[0] == 123
            #
            # # Number of drivers
            # num_drivers = struct.unpack('<I', f.read(4))[0]
            # print(f"Found {num_drivers} drivers")
            # # for driver_id in range(num_drivers):
            # #     drivers_meta.append(parse_driver(f))

            # Define the regex pattern
            pattern = re.compile(b"\x3E\xE1\x7A\xB4\x3E..\x00\x00")
            f.seek(0)
            data = f.read()
            # Iterate all matches and get match end position
            for idx, match in enumerate(pattern.finditer(data)):

                if idx == 0 and skip_start:
                    continue

                car = CarData()

                print(f"Found at position {match.end()}")
                f.seek(match.end() - 4)

                # Extra data
                extra_data = {"is_extended": is_extended}

                max_items = 60000

                # Get the number of elements in the packet list
                num_items = struct.unpack('<I', f.read(4))[0]
                if num_items > max_items:
                    raise ValueError(f"Number of items is too high {num_items}")
                # Split data 104 byte chunks
                for i in range(0, num_items):
                    car._packet_data_104.append(Packet104(f, **extra_data))

                # Get the number of elements in the next packet list
                num_items = struct.unpack('<I', f.read(4))[0]
                if num_items > max_items:
                    raise ValueError(f"Number of items is too high {num_items}")
                for i in range(0, num_items):
                    # car_data_2.append(parse_48_packet(f))
                    car._packet_data_48.append(Packet48(f, **extra_data))

                # Get the number of elements in the next packet list
                num_items = struct.unpack('<I', f.read(4))[0]
                if num_items > max_items:
                    raise ValueError(f"Number of items is too high {num_items}")
                for i in range(0, num_items):
                    # car_data_3.append(parse_72_packet(f))
                    car._packet_data_72.append(Packet72(f, **extra_data))

                # Get the number of elements in the next packet list
                num_items = struct.unpack('<I', f.read(4))[0]
                if num_items > max_items:
                    raise ValueError(f"Number of items is too high {num_items}")
                for i in range(0, num_items):
                    # car_data_4.append(parse_24_packet(f))
                    car._packet_data_24.append(Packet24(f, **extra_data))

                # Get the number of elements in the next packet list
                num_items = struct.unpack('<I', f.read(4))[0]
                if num_items > max_items:
                    raise ValueError(f"Number of items is too high {num_items}")
                for i in range(0, num_items):
                    # car_data_5.append(parse_16_packet(f))
                    car._packet_data_16.append(Packet16(f, **extra_data))

                self._cars.append(car)


    def build_coord_data(self):
        # Build the data
        data = [
            {key: item for key, item in car_obj.cast_coord_info().items()}
            for car_id, car_obj in enumerate(self._cars)
        ]
        # data = deep_merge(data)

        # # Order the dictionary by keys
        # data = dict(sorted(data.items()))
        # For each car, sort the dictionary
        data = [
            dict(sorted(item.items()))
            for item in data
        ]

        data_keys = [np.array(list(item.keys())) for item in data]
        data_values = [np.array(list(item.values())) for item in data]

        start_time = int(min([
            car_data_keys.min()
            for car_data_keys in data_keys
            if car_data_keys.shape[0] > 0
        ]))
        end_time = int(max([
            car_data_keys.max()
            for car_data_keys in data_keys
            if car_data_keys.shape[0] > 0
        ]))
        increment = 0.1

        data_names = ["coords", "contactPoint1", "contactPoint2", "contactPoint3", "contactPoint4"]
        n, c = (int(end_time) - int(start_time)) * int(1 / increment), 3

        # Key map, for each resampled location, we store the nearest prev, next data index for each car.
        cars_key_map = (np.zeros((n, len(self._cars), 2)) - 1).astype(np.int32)

        class PeekIter:
            def __init__(self, lst):
                self.lst = lst
                self.idx = 0

            def __iter__(self):
                return self

            def peek(self):
                if self.idx >= len(self.lst):
                    return None
                else:
                    return self.lst[self.idx]

            def rewind(self):
                self.idx-= 1

            def __next__(self):
                if self.idx < len(self.lst):
                    item = self.lst[self.idx]
                    self.idx += 1
                    return item
                raise StopIteration

        data_key_iters = [
            PeekIter(list(keys))
            for keys in data_keys
        ]
        # For each time index in the resampled data
        for i in range(0, n):
            time_idx = i * increment + start_time
            # For each time index in the existing data
            for car_id, data_key_iter in enumerate(data_key_iters):
                for data_key in data_key_iter:
                    data_key_idx = data_key_iter.idx - 1
                    if data_key is None:
                        raise Exception("Well fuck.")
                    elif data_key >= time_idx:
                        cars_key_map[i, car_id, 1] = data_key_idx
                        data_key_iter.rewind()
                        break
                    else:
                        peek = data_key_iter.peek()
                        if peek is not None and (data_key < peek < time_idx):
                            continue
                        cars_key_map[i, car_id, 0] = data_key_idx

        # We now have the prev and next index for each resample point.
        # We can now interpolate!
        resampled_data = np.zeros((n, len(self._cars), len(data_names), c))
        cars_key_map = cars_key_map.transpose(1, 0, 2)
        invalids = np.zeros((n, len(self._cars)))
        for car_id in range(cars_key_map.shape[0]):
            if data_values[car_id].shape[0] == 0:
                continue
            car_map = cars_key_map[car_id]
            car_mask = ~(car_map == -1).astype(np.bool).any(-1)
            car_map = car_map[car_mask]
            # Index the data_values
            left_values = data_values[car_id][car_map[:, 0]]
            right_values = data_values[car_id][car_map[:, 1]]
            left_time_idx = data_keys[car_id][car_map[:, 0]]
            right_time_idx = data_keys[car_id][car_map[:, 1]]
            time_idx = (np.arange(0, n) * increment)[car_mask] + start_time
            # Get alpha ratio
            alpha = (time_idx - left_time_idx) / (right_time_idx - left_time_idx)
            resampled_data[car_mask, car_id] = left_values + alpha[:, None, None] * (right_values - left_values)
            invalids[:, car_id] = ~car_mask

        # Compare consecutive data
        diff_mask = ((resampled_data[1:, :, 1:].round(3) - resampled_data[0:-1, :, 1:].round(3)) == 0).any(2).any(-1)
        diff_mask = np.concatenate([np.ones(diff_mask[0:1].shape), diff_mask], axis=0).astype(np.int32)
        invalids = diff_mask.astype(np.bool) | invalids.astype(np.bool)

        return {
            "data": resampled_data,
            "invalids": invalids
        }


unknown_type = "<f"
