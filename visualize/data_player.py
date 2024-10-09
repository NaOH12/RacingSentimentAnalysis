import os

import numpy as np
import cv2

from renderer import Box, Colour, Sphere, Camera, SceneViewer


class ReplayPlayer:
    def __init__(self, file_id=None, data_dir=None, track_dir=None):
        self._data_dir = data_dir
        self._track_dir = track_dir

        # Load race data
        data_file = f"{data_dir}{file_id}.npy"
        self._data = np.load(data_file, allow_pickle=True).item()

        # Load track data
        track_id = self._data['track_id']
        track_file = f"{track_dir}{track_id}.npy"
        if os.path.exists(track_file):
            self._track = np.load(track_file, allow_pickle=True).item()
        else:
            self._track = None


    @classmethod
    def _move_camera(cls, camera, key):
        movement_speed = 100.0
        rotation_speed = 0.1

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
            camera.focal_length += 1
            print("Focal Length: ", camera.focal_length)
        elif key == ord('v'):
            camera.focal_length -= 1
            print("Focal Length: ", camera.focal_length)

    def render(self, car_focus_id):
        # OpenCV 3D rendering of the data
        # Iterate over the packet ids (as timesteps)
        # Render the track points (as sphere circle points)
        # and the car contactPoints (as vertices of the rectangle base of a 3D box of height 10)

        # Create window
        window_name = "3D Rendering"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_size = (1600, 800)
        cv2.resizeWindow(window_name, *window_size)

        # # Create track points
        # track_points = self._build_track()

        # Session data {time_id: {car_id: contactPoint_id: np array of shape (3,)}}
        # session_data = self._build_data()

        samples = self._data['data']
        invalids = self._data['invalids']
        track_points = np.concatenate([self._track['left_track'], self._track['right_track']], axis=0)
        racing_line_points = self._track['racing_line']
        num_samples, num_cars, c, _ = samples.shape

        car_objects = [
            Box(height=2, colour=Colour.pink())
            for _ in range(num_cars)
        ]
        track_point_spheres = [
            Sphere(radius=1, point=point, colour=Colour.dark_blue())
            for point in track_points
        ]
        racing_line_spheres  = [
            Sphere(radius=1, point=point, colour=Colour.green())
            for point in racing_line_points
        ]
        # racing_line_spheres = []

        distance = 10
        camera = Camera(
            window_size=window_size, focal_length=200,
            rotation=np.array([0, 0, 0]).astype(np.float32),
            translation=np.array([0, 0, 0]).astype(np.float32)
        )

        graphs = []

        scene_viewer = SceneViewer(
            objects=car_objects + track_point_spheres + racing_line_spheres + graphs,
            camera=camera,
            background_colour=Colour.cream(),
            max_render_distance=50
        )

        count = 0
        focus_point = np.array([0, 0, 0])
        direction = np.array([0, 0, 1])

        # Iterate over packet IDs (timesteps)
        for idx, (data_sample, sample_invalids) in enumerate(zip(samples, invalids)):

            # Skip invalid data
            if sample_invalids[car_focus_id] == True:
                continue

            # Update the car objects with the contact points
            for car_id, car_data in enumerate(data_sample):
                # Wait (delay) 0.1 seconds
                # cv2.waitKey(1)
                car_object = car_objects[car_id]
                # Get the contact points
                coords = car_data[0]
                # contact_points = car_data[2:]
                contact_points = np.concatenate([
                    # car_data[2:3], car_data[3:4], car_data[4:5], car_data[5:6]
                    car_data[1:2], car_data[2:3], car_data[4:5], car_data[3:4]
                ])
                # Update the base cvcorners of the car object
                car_object._base_corners = contact_points

                # Compute the car direction by comparing the first and third contact points
                car_direction = contact_points[2] - contact_points[0]
                car_direction = car_direction / np.linalg.norm(car_direction)

                if car_id == car_focus_id:
                    focus_point = coords
                    direction = car_direction

                    graph_graphs = []
                    for graph in graphs:
                        graph_graphs = graph_graphs + graph.graphs

                    # Update the graph
                    for graph in graph_graphs:
                        if graph.label in car_data:
                            new_data = car_data[graph.label]

                            # If new_data is nan then set to 0
                            if np.isnan(new_data):
                                new_data = np.array([0])

                            if isinstance(new_data, np.ndarray):
                                graph.data = np.concatenate((graph.data, car_data[graph.label]))[-100:]
                            else:
                                graph.data = np.concatenate((graph.data, np.array([car_data[graph.label]])))[-100:]
                            # Remove nan values
                            graph.data = graph.data[~np.isnan(graph.data)]
                        else:
                            # Concat last value from graph.data
                            graph.data = np.concatenate((graph.data, graph.data[-1:]))[-100:]

            # Focus on the car object
            camera.focus(point=focus_point, direction=direction, distance=distance)

            # Render the scene
            canvas = scene_viewer.render()

            # Display the rendered image in the window
            cv2.imshow(window_name, canvas)

            # Wait
            key = cv2.waitKey(1)
            self._move_camera(camera, key)

            # Change the distance of the camera
            if key == ord('z'):
                distance += 100
                print("Distance: ", distance)
            elif key == ord('x'):
                distance -= 100
                print("Distance: ", distance)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # file_name = "data/16209.npy"
    data_dir = 'C:\\Users\\noahl\Documents\ACCDataset/race_data/'
    # data_dir = "datasets/builder/ghost_data/"
    track_dir = 'C:\\Users\\noahl\Documents\ACCDataset/track_data/'
    race_data = ReplayPlayer(file_id='10006', data_dir=data_dir, track_dir=track_dir)

    race_data.render(car_focus_id=5)
