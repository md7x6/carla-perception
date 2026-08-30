import carla
import numpy as np

def spawn_camera(world, blueprints, ego_vehicle, width, height, fov, transform):
    camera_blueprint = blueprints.find("sensor.camera.rgb")
    camera_blueprint.set_attribute("image_size_x", str(width))
    camera_blueprint.set_attribute("image_size_y", str(height))
    camera_blueprint.set_attribute("fov", str(fov))

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=ego_vehicle)

    print("Camera spawned:", camera.id)
    return camera

class FrameBuffer:
    """Holds the most recent camera frame, converted from CARLA's raw BGRA buffer."""

    def __init__(self):
        self.latest_frame = None
        self.latest_frame_id = None 

    def on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_frame = array[:, :, :3].copy()  # drop alpha channel
        self.latest_frame_id = image.frame