import copy

import cv2
import numpy as np
from matplotlib import colors


class Camera(object):
    def __init__(
            self, window_size: tuple = None, focal_length: float = 25,
            rotation: np.ndarray = None, translation: np.ndarray = None
    ):
        assert window_size is None or len(window_size) == 2, "Window size must be a tuple of 2 integers"
        self.window_size = window_size
        self.focal_length = focal_length
        self._set_intrinsic()
        self.rotation = rotation if rotation is not None else np.zeros(3).astype(np.float32)
        self.translation = translation if translation is not None else np.zeros(3).astype(np.float32)

    def _set_intrinsic(self):
        self.intrinsic = np.array([
            [self.focal_length, 0, self.window_size[0] // 2],
            [0, self.focal_length, self.window_size[1] // 2],
            [0, 0, 1]
        ])

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = value
        self._set_intrinsic()

    def translate(self, translation: np.ndarray):
        self.translation += translation.astype(np.float32)

    def rotate(self, rotation: np.ndarray):
        self.rotation += rotation.astype(np.float32)

    def project_points(self, points: np.ndarray):
        # Bring points to camera space via translation and rotation
        points -= copy.deepcopy(self.translation)
        # Rotate points
        points = np.dot(points, cv2.Rodrigues(self.rotation)[0].T)
        # Is the point in front of the camera?
        if np.any(points[:, 2] < 0):
            return None
        # Use OpenCV projectPoints function
        projected_points = cv2.projectPoints(
            points.astype(np.float32),
            copy.deepcopy(self.rotation).astype(np.float32),
            # copy.deepcopy(self.translation).astype(np.float32),
            np.zeros(3).astype(np.float32),
            self.intrinsic.astype(np.float32),
            None
        )[0]
        projected_points = np.squeeze(projected_points, axis=1)
        return projected_points

    def get_distance(self, point: np.ndarray):
        return np.linalg.norm(copy.deepcopy(self.translation) - point)

    def get_square_distance(self, point: np.ndarray):
        return ((copy.deepcopy(self.translation) - point) ** 2).sum()

    def focus(self, point: np.ndarray, direction: np.ndarray, distance: float):
        """
        Lock the camera behind an object
        :param point: point to focus on
        :param distance:
        :return:
        """
        self.translation[0] = (point[0] - distance * direction[0])
        self.translation[1] = (point[1] - distance * direction[1]) - 20
        self.translation[2] = (point[2] - distance * direction[1]) - 10
        self.rotation = np.array([0.4, 0, 0])

    def _look_at(self, point: np.ndarray):
        """
        Look at a point
        :param point:
        :return:
        """
        self.rotation = point - self.translation

    def draw(self, canvas: np.ndarray, camera):
        projected_point = camera.project_points(copy.deepcopy(self.translation))[0, 0]
        if projected_point is None:
            return
        # Draw camera
        cv2.circle(canvas, projected_point.astype(np.int16), 5, Colour.green(), 3)

        length = 100
        # Define axis points
        axis_points = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ]).astype(np.float32) + copy.deepcopy(self.translation)
        # Project points
        projected_points = camera.project_points(copy.deepcopy(axis_points))
        if projected_points is None:
            return
        # Draw lines between points
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[1].astype(np.int16)),
                 (0, 0, 255), 3)
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[2].astype(np.int16)),
                 (0, 255, 0), 3)
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[3].astype(np.int16)),
                 (255, 0, 0), 3)


class Object:
    def draw(self, canvas: np.ndarray, camera: Camera):
        raise NotImplementedError


class Axis(Object):
    def __init__(self, length: float = 100, position: np.ndarray = None):
        self.length = length
        self.position = position if position is not None else np.zeros(3).astype(np.float32)

    def draw(self, canvas: np.ndarray, camera: Camera):
        # Define axis points
        axis_points = np.array([
            [0, 0, 0],
            [self.length, 0, 0],
            [0, self.length, 0],
            [0, 0, self.length]
        ]).astype(np.float32) + self.position
        # Project points
        projected_points = camera.project_points(axis_points)
        if projected_points is None:
            return
        # Draw lines between points
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[1].astype(np.int16)),
                 (0, 0, 255), 3)
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[2].astype(np.int16)),
                 (0, 255, 0), 3)
        cv2.line(canvas, tuple(projected_points[0].astype(np.int16)), tuple(projected_points[3].astype(np.int16)),
                 (255, 0, 0), 3)


class Colour(tuple[3]):

    def __new__(cls, r: int, g: int, b: int):
        return super(Colour, cls).__new__(cls, (r, g, b))

    def __init__(self, r: int, g: int, b: int):
        assert 0 <= r <= 255, "Red channel must be between 0 and 255"
        assert 0 <= g <= 255, "Green channel must be between 0 and 255"
        assert 0 <= b <= 255, "Blue channel must be between 0 and 255"
        self.r = r
        self.g = g
        self.b = b

    def __str__(self):
        return f"({self.r}, {self.g}, {self.b})"

    @classmethod
    def from_hex(cls, hex_str: str):
        assert len(hex_str) == 6, "Hex string must be in the format RRGGBB"
        rgb = colors.to_rgb(f"#{hex_str}")
        return cls(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    @classmethod
    def green(cls):
        return cls.from_hex("6D9F71")

    @classmethod
    def cream(cls):
        return cls.from_hex("E4E3D3")

    @classmethod
    def dark_blue(cls):
        return cls.from_hex("2E2D4D")

    @classmethod
    def pink(cls):
        return cls.from_hex("CB48B7")

    @classmethod
    def red(cls):
        return cls.from_hex("FF0000")

    @classmethod
    def blue(cls):
        return cls.from_hex("0000FF")

    @classmethod
    def grey(cls):
        return cls.from_hex("808080")

    @classmethod
    def dark_grey(cls):
        return cls.from_hex("A9A9A9")


class Box(Object):
    def __init__(self, height: float, base_corners: np.ndarray = None, colour: Colour = Colour.green()):
        self._base_corners = base_corners
        self._height = height
        self._colour = colour

    def _get_box_corners(self):
        return np.concatenate((
            self._base_corners,
            self._base_corners + np.array([0, -self._height, 0])
        ), axis=0)

    @property
    def position(self):
        return self._base_corners.mean(0)

    def draw(self, canvas: np.ndarray, camera: Camera):
        box_corners = self._get_box_corners()
        projected_corners = camera.project_points(copy.deepcopy(box_corners))
        if projected_corners is None:
            return
        # Draw circles at projected corners
        for corner in projected_corners[:4]:
            cv2.circle(canvas, tuple(corner.astype(np.int16)), 3, self._colour, 1)
        # Draw lines between corners
        # Lines (0, 1), (1, 2), (2, 3), (3, 0) - Base
        # Lines (0, 4), (1, 5), (2, 6), (3, 7) - Base to top
        # Lines (4, 5), (5, 6), (6, 7), (7, 4) - Top
        for i in range(4):
            cv2.line(canvas, tuple(projected_corners[i].astype(np.int16)),
                     tuple(projected_corners[(i + 1) % 4].astype(np.int16)),
                     self._colour, 1)
            cv2.line(canvas, tuple(projected_corners[i].astype(np.int16)),
                     tuple(projected_corners[i + 4].astype(np.int16)),
                     self._colour, 1)
            cv2.line(canvas, tuple(projected_corners[i + 4].astype(np.int16)),
                     tuple(projected_corners[(i + 1) % 4 + 4].astype(np.int16)),
                     self._colour, 1)


class Sphere(Object):
    def __init__(self, radius: float, point: np.ndarray = None, colour: Colour = Colour.green()):
        assert radius > 0, "Radius must be greater than 0"
        self.point = point
        self.radius = radius
        self._colour = colour

    @property
    def position(self):
        return self.point

    def draw(self, canvas: np.ndarray, camera: Camera):
        projected_point = camera.project_points(copy.deepcopy(np.expand_dims(self.point, axis=0)))
        if projected_point is None:
            return
        cv2.circle(canvas, tuple((projected_point.astype(np.int16))[0]), int(self.radius), self._colour, -1)


class Line(Object):
    def __init__(self, colour: Colour = Colour.green(), points: np.ndarray = None):
        self.points = points
        self._colour = colour

    @property
    def position(self):
        if self.points is None:
            return np.zeros(3)
        return self.points.mean(0)

    def draw(self, canvas: np.ndarray, camera: Camera):
        if self.points is None:
            return
        projected_points = camera.project_points(copy.deepcopy(self.points))
        if projected_points is None:
            return
        for i in range(len(projected_points) - 1):
            cv2.line(
                canvas, tuple(projected_points[i].astype(np.int16)), tuple(projected_points[i + 1].astype(np.int16)),
                self._colour, 1
            )


class GraphArray(Object):

    def __init__(
            self,
            top_left: tuple, width: int, height: int,
            graph_labels: list[str], colour: Colour = Colour.green(), background_colour: Colour = Colour.cream(),
            contract_axis=None
    ):
        gap = 10
        total_gap_width = (len(graph_labels) - 1) * gap
        usable_window_width = width - total_gap_width
        graph_width = usable_window_width // len(graph_labels)

        # If contract_axis is a boolean, convert to list
        if isinstance(contract_axis, bool):
            contract_axis = [contract_axis] * len(graph_labels)

        self.graphs = [
            Graph(
                top_left=(
                    top_left[0] + ((graph_width + gap) * i),
                    top_left[1]
                ),
                size=(graph_width, height),
                colour=colour,
                background_colour=background_colour,
                label=graph_labels[i],
                contract_axis=contract_axis[i] if contract_axis is not None else False
            )
            for i in range(len(graph_labels))
        ]

    def reset_data(self):
        for graph in self.graphs:
            graph.reset_data()

    def draw(self, canvas: np.ndarray, camera: Camera):
        for graph in self.graphs:
            try:
                graph.draw(canvas, camera)
            except Exception as e:
                print(e)


class Graph(Object):
    MAX_DATA_POINTS = 100

    def __init__(
            self,
            top_left: tuple = None, size: tuple = None,
            data: np.ndarray = None, colour: Colour = Colour.green(), background_colour: Colour = Colour.cream(),
            label: str = None, contract_axis: bool = False
    ):
        self._data = data if data is not None else np.array([])
        self._top_left = top_left if top_left is not None else np.array([0, 0])
        assert size is None or len(size) == 2, "Size must be a tuple of 2 integers"
        assert size[0] > 0 and size[1] > 0, "Size must be greater than 0"
        self._size = size if size is not None else (100, 100)
        self._colour = colour
        self.label = label if label is not None else None
        self._min_val = None
        self._max_val = None
        self._contract_axis = contract_axis
        self._background_colour = background_colour

    @property
    def data(self):
        return self._data

    def reset_data(self):
        self._data = np.array([])

    @data.setter
    def data(self, value):
        assert len(value.shape) == 1, "Data must be a 1D array"
        assert len(value) <= self.MAX_DATA_POINTS, "Data must have a maximum of 100 elements"
        self._data = value

    def draw(self, canvas: np.ndarray, camera: Camera):
        if len(self.data) == 0:
            return
        # Ensure top_left is in canvas
        assert 0 <= self._top_left[0] < camera.window_size[0], "Top left x must be within canvas"
        assert 0 <= self._top_left[1] < camera.window_size[1], "Top left y must be within canvas"
        # Ensure size is within canvas
        assert self._size[0] + self._top_left[0] < camera.window_size[0], "Graph width must be within canvas"
        assert self._size[1] + self._top_left[1] < camera.window_size[1], "Graph height must be within canvas"

        # Draw background
        cv2.rectangle(
            canvas, self._top_left, (self._top_left[0] + self._size[0], self._top_left[1] + self._size[1]),
            self._background_colour, -1
        )

        # Minimum and maximum values
        local_min_val = np.min(self.data)
        local_max_val = np.max(self.data)
        if self._min_val is None or local_min_val < self._min_val:
            self._min_val = local_min_val
        if self._max_val is None or local_max_val > self._max_val:
            self._max_val = local_max_val
        # Convert to scientific notation string
        if self._contract_axis is False:
            min_val_str = "{:.2e}".format(self._min_val)
            max_val_str = "{:.2e}".format(self._max_val)
            # Normalize the data using min and max values with safe division
            data = (self.data - self._min_val) / (self._max_val - self._min_val + (self._min_val / 1000))
        else:
            min_val_str = "{:.2e}".format(local_min_val)
            max_val_str = "{:.2e}".format(local_max_val)
            # Normalize the data using min and max values np.linalg.norm
            data = (self.data - local_min_val) / (local_max_val - local_min_val + (local_min_val / 1000))

        # Add label
        if self.label is not None:
            cv2.putText(
                canvas, self.label, (self._top_left[0], self._top_left[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._colour, 1
            )

        # Draw axis
        # Place min and max values on the graph
        cv2.putText(
            canvas, min_val_str, (self._top_left[0], self._top_left[1] + self._size[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, self._colour, 1
        )
        cv2.putText(
            canvas, max_val_str, (self._top_left[0], self._top_left[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, self._colour, 1
        )
        draw_area_top_left = (self._top_left[0], self._top_left[1] + 15)
        draw_size = (self._size[0], self._size[1] - 30)
        # If there exists the value 0 between min and max, draw a horizontal line
        if self._min_val < 0 < self._max_val:
            # Ratio of 0 between min and max
            if self._contract_axis:
                zero_ratio = (0 - local_min_val) / (local_max_val - local_min_val)
            else:
                zero_ratio = (0 - self._min_val) / (self._max_val - self._min_val)
            # Protect against nan
            if np.isnan(zero_ratio):
                zero_ratio = 0
            y_pos = int(draw_area_top_left[1] + draw_size[1] - zero_ratio * draw_size[1])
            cv2.line(
                canvas,
                (draw_area_top_left[0], y_pos),
                (draw_area_top_left[0] + draw_size[0], y_pos),
                self._colour, 1
            )

        # Draw graph
        old_x = None
        old_y = None
        for i in range(0, len(self.data)):
            x = draw_area_top_left[0] + (i * (draw_size[0] / self.MAX_DATA_POINTS))
            y = draw_area_top_left[1] + draw_size[1] - (data[i] * draw_size[1])
            if i > 0:
                try:
                    # Draw line
                    cv2.line(
                        canvas,
                        (int(old_x), int(old_y)),
                        (int(x), int(y)),
                        self._colour, 1
                    )
                except Exception as e:
                    print(self.label, e)
            old_x = x
            old_y = y


class SceneViewer:
    def __init__(
            self,
            objects: list[Object] = None,
            camera: Camera = None,
            background_colour: Colour = Colour.cream(),
            max_render_distance=None
    ):
        self._objects = objects if objects is not None else []
        self._camera = camera
        self._background_colour = background_colour
        self._max_render_distance = max_render_distance
        self._square_max_render_distance = None if max_render_distance is None else max_render_distance ** 2

    def render(self):
        # Create canvas with colour
        canvas = np.zeros((self._camera.window_size[1], self._camera.window_size[0], 3), dtype=np.uint8)
        canvas[:, :] = self._background_colour

        # Draw objects
        if self._max_render_distance is None:
            for obj in self._objects:
                obj.draw(canvas=canvas, camera=self._camera)
        else:
            for obj in self._objects:
                square_distance = self._camera.get_square_distance(obj.position)
                if square_distance <= self._square_max_render_distance:
                    obj.draw(canvas=canvas, camera=self._camera)

        return canvas


# If main exec
if __name__ == "__main__":
    # visualize
    camera = Camera(
        window_size=(800, 800),
        focal_length=0.9,
        translation=np.array([0, 0, 0]).astype(np.float32),
        rotation=np.array([0, 0, 0]).astype(np.float32)
    )
    objects = [
        camera
    ]

    movement_speed = 100.0
    rotation_speed = 0.1

    while True:

        key = cv2.waitKey(1)
        if key == ord('w'):
            camera.translate(np.array([0, 0, 1]) * movement_speed)
        elif key == ord('s'):
            camera.translate(np.array([0, 0, -1]) * movement_speed)
        elif key == ord('a'):
            camera.translate(np.array([-1, 0, 0]) * movement_speed)
        elif key == ord('d'):
            camera.translate(np.array([1, 0, 0]) * movement_speed)
        elif key == ord('q'):
            camera.translate(np.array([0, 1, 0]) * movement_speed)
        elif key == ord('e'):
            camera.translate(np.array([0, -1, 0]) * movement_speed)

        # Rotate the camera
        if key == ord('i'):
            camera.rotate(np.array([1, 0, 0]) * rotation_speed)
        elif key == ord('k'):
            camera.rotate(np.array([-1, 0, 0]) * rotation_speed)
        elif key == ord('j'):
            camera.rotate(np.array([0, 1, 0]) * rotation_speed)
        elif key == ord('l'):
            camera.rotate(np.array([0, -1, 0]) * rotation_speed)
        elif key == ord('u'):
            camera.rotate(np.array([0, 0, 1]) * rotation_speed)
        elif key == ord('o'):
            camera.rotate(np.array([0, 0, -1]) * rotation_speed)

        # Change the focal length of the camera
        elif key == ord('c'):
            camera.focal_length += 100
            print("Focal Length: ", camera.focal_length)
        elif key == ord('v'):
            camera.focal_length -= 100
            print("Focal Length: ", camera.focal_length)

        canvas = SceneViewer(objects=objects, camera=camera).render()
        cv2.imshow("Scene", canvas)
