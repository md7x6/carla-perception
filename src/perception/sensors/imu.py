
import carla
import numpy as np


class IMUFrameBuffer:
    """Holds the most recent CARLA IMU measurements."""

    def __init__(self):
        self.latest_accelerometer = None
        self.latest_gyroscope = None
        self.latest_compass = None
        self.latest_frame_id = None

        self.samples_received = 0 
    def on_imu(self, imu_data):
        """
        Store CARLA IMU measurements.

        Accelerometer: ax, ay, az (m/s²)
        Gyroscope: gx, gy, gz (rad/s)
        Compass: heading (radians)
        """
        self.samples_received += 1

        # Ignore initial samples while the simulation settles.
        if self.samples_received <= 10:
            return

        self.latest_accelerometer = np.array([
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z
        ], dtype=np.float32)

        self.latest_gyroscope = np.array([
            imu_data.gyroscope.x,
            imu_data.gyroscope.y,
            imu_data.gyroscope.z
        ], dtype=np.float32)

        self.latest_compass = float(imu_data.compass)
        self.latest_frame_id = imu_data.frame


def spawn_imu(
    world,
    blueprints,
    ego_vehicle,
    transform=None,
    sensor_tick=0.0,
):
    imu_blueprint = blueprints.find("sensor.other.imu")

    imu_blueprint.set_attribute(
        "sensor_tick",
        str(sensor_tick)
    )

    if transform is None:
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

    imu = world.spawn_actor(
        imu_blueprint,
        transform,
        attach_to=ego_vehicle
    )

    print("IMU spawned:", imu.id)

    return imu
