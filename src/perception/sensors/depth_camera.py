import carla
import numpy as np

class DepthFrameBuffer:
    """Holds the most recent CARLA depth frame."""

    def __init__(self):
        self.latest_depth = None
        self.latest_frame_id = None

    def on_image(self, image):
        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        ).reshape((image.height, image.width, 4))

        # raw_data is BGRA in memory
        b = array[:, :, 0].astype(np.float32)
        g = array[:, :, 1].astype(np.float32)
        r = array[:, :, 2].astype(np.float32)

        # CARLA depth encoding: R = LSB, G = middle, B = MSB
        depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)

        # Normalized [0,1] represents 0-1000 meters
        depth *= 1000.0

        self.latest_depth = depth
        self.latest_frame_id = image.frame
        """
        print(
            f"Depth frame={image.frame}, "
            f"min={depth.min():.2f}, "
            f"max={depth.max():.2f}, "
            f"center={depth[image.height // 2, image.width // 2]:.2f}"
        )"""

    def get_depth_at_pixel(self, depth_image, x, y):
        return depth_image[y, x]


def spawn_depth_camera(
    world,
    blueprints,
    ego_vehicle,
    width,
    height,
    fov,
    transform=None
):
    depth_blueprint = blueprints.find("sensor.camera.depth")

    depth_blueprint.set_attribute(
        "image_size_x",
        str(width)
    )

    depth_blueprint.set_attribute(
        "image_size_y",
        str(height)
    )

    depth_blueprint.set_attribute(
        "fov",
        str(fov)
    )

    # IMPORTANT:
    # Use exactly the same transform as the RGB camera so RGB and
    # depth pixels correspond. Previously this was hardcoded here,
    # which silently breaks alignment if the RGB camera's transform
    # ever changes elsewhere. Caller should pass the RGB camera's
    # actual transform in.
    if transform is None:
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

    depth_camera = world.spawn_actor(
        depth_blueprint,
        transform,
        attach_to=ego_vehicle
    )

    print("Depth camera spawned:", depth_camera.id)

    return depth_camera