# perception/sensors/lidar.py

import carla
import numpy as np


class LiDARFrameBuffer:
    """Holds the most recent CARLA LiDAR point cloud."""

    def __init__(self):
        self.latest_points = None
        self.latest_frame_id = None

    def on_lidar(self, point_cloud):
        """
        Convert CARLA LiDAR raw data into an Nx4 NumPy array.

        Columns:
            x, y, z, intensity
        """

        points = np.frombuffer(
            point_cloud.raw_data,
            dtype=np.float32
        )

        points = points.reshape((-1, 4))

        self.latest_points = points.copy()
        self.latest_frame_id = point_cloud.frame


def spawn_lidar(
    world,
    blueprints,
    ego_vehicle,
    transform=None,
    range=50.0,
    channels=32,
    points_per_second=56000,
    rotation_frequency=20.0,
    upper_fov=10.0,
    lower_fov=-30.0,
):
    lidar_blueprint = blueprints.find("sensor.lidar.ray_cast")

    lidar_blueprint.set_attribute(
        "range",
        str(range)
    )

    lidar_blueprint.set_attribute(
        "channels",
        str(channels)
    )

    lidar_blueprint.set_attribute(
        "points_per_second",
        str(points_per_second)
    )

    lidar_blueprint.set_attribute(
        "rotation_frequency",
        str(rotation_frequency)
    )

    lidar_blueprint.set_attribute(
        "upper_fov",
        str(upper_fov)
    )

    lidar_blueprint.set_attribute(
        "lower_fov",
        str(lower_fov)
    )

    if transform is None:
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

    lidar = world.spawn_actor(
        lidar_blueprint,
        transform,
        attach_to=ego_vehicle
    )

    print("LiDAR spawned:", lidar.id)

    return lidar