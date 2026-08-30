import cv2
import random
import time
import numpy as np
import carla

from perception.utils.config import load_config
from perception.simulation.connection import connect
from perception.simulation.traffic import setup_traffic_manager
from perception.simulation.ego import spawn_ego_vehicle
from perception.simulation.traffic import spawn_random_traffic
from perception.simulation.traffic import spawn_pedestrians
from perception.sensors.camera import spawn_camera, FrameBuffer
from perception.sensors.depth_camera import DepthFrameBuffer, spawn_depth_camera

from perception.ground_truth.actors import get_target_actors, get_actor_class
from perception.ground_truth.projection import build_camera_matrix, project_bbox, bbox_2d_from_points
from perception.ground_truth.visibility import is_bbox_visible

from perception.ground_truth.visualization import draw_bbox
#from save_dataset import save_dataset


def wait_for_synced_frame(world, frame_buffer, depth_buffer, max_ticks=10, timeout=1.0):
    """
    Tick until both RGB and depth buffers have delivered a frame
    matching the current simulation frame.

    Sensor callbacks are asynchronous relative to tick() — a single
    tick() does not guarantee fresh, matched data has arrived yet.
    Confirmed necessary: without this, frame_buffer/depth_buffer
    were observed to still be None immediately after tick().
    """
    for _ in range(max_ticks):
        world.tick()
        current_frame = world.get_snapshot().frame

        deadline = time.time() + timeout
        while (frame_buffer.latest_frame_id != current_frame
               or depth_buffer.latest_frame_id != current_frame):
            if time.time() > deadline:
                break
            time.sleep(0.005)

        if (frame_buffer.latest_frame_id == current_frame
                and depth_buffer.latest_frame_id == current_frame):
            return current_frame

    return None


def camera_settings_wh(camera):
    # Pull live attributes back from the sensor rather than trusting
    # a possibly-stale config dict.
    attrs = camera.attributes
    return int(attrs["image_size_x"]), int(attrs["image_size_y"])


def run_perception_loop(
    world,
    ego_vehicle,
    camera,
    depth_camera,
    frame_buffer,
    depth_buffer,
    K,
    window_name="Carla Perception",
):
    width, height = camera_settings_wh(camera)

    while True:
        current_frame = wait_for_synced_frame(world, frame_buffer, depth_buffer)

        if current_frame is None:
            print("WARNING: RGB/depth did not sync this step, skipping frame.")
            continue

        rgb = frame_buffer.latest_frame
        depth = depth_buffer.latest_depth

        if rgb is None or depth is None:
            continue

        annotated = rgb.copy()

        for actor in get_target_actors(world):
            if actor.id == ego_vehicle.id:
                continue

            pixels = project_bbox(actor, camera, K)
            bbox = bbox_2d_from_points(pixels, width, height)

            if bbox is None:
                continue  # outside camera view / fully behind camera

            visible, ratio = is_bbox_visible(
                actor=actor,
                camera=camera,
                depth_image=depth,
                bbox=bbox,
            )

            if not visible:
                continue  # partially/fully occluded below threshold — reject

            label = f"{get_actor_class(actor)} {ratio:.2f}"
            annotated = draw_bbox(annotated, bbox, label=label)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

import os


def save_dataset(
    world,
    ego_vehicle,
    camera,
    depth_camera,
    frame_buffer,
    depth_buffer,
    K,
    output_dir,
    num_images=5,
    save_interval=22,
):
    """
    Capture synchronized RGB frames + YOLO-format labels.

    For each saved frame:
        - images/frame_XXXXXX.png   (RGB)
        - labels/frame_XXXXXX.txt   (YOLO format: class cx cy w h, normalized)

    Only actors passing is_bbox_visible() are written to the label file.
    Uses wait_for_synced_frame() so RGB and depth are guaranteed to be
    from the same simulation tick before any projection/visibility work
    is done — this was the root cause of earlier depth misalignment.
    """

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Fixed class list -> index mapping for YOLO format.
    # Extend this if get_actor_class() ever returns more classes.
    class_to_id = {
        "vehicle": 0,
        "pedestrian": 1,
    }

    width, height = camera_settings_wh(camera)

    saved_count = 0
    attempted_frames = 0

    while saved_count < num_images:
        current_frame = wait_for_synced_frame(world, frame_buffer, depth_buffer)

        if current_frame is None:
            print("WARNING: RGB/depth did not sync this step, skipping.")
            continue

        attempted_frames += 1

        if current_frame % save_interval != 0:
            continue

        rgb = frame_buffer.latest_frame
        depth = depth_buffer.latest_depth

        if rgb is None or depth is None:
            continue

        # ---- Build YOLO labels for this frame ----
        label_lines = []

        for actor in get_target_actors(world):
            if actor.id == ego_vehicle.id:
                continue

            pixels = project_bbox(actor, camera, K)
            bbox = bbox_2d_from_points(pixels, width, height)

            if bbox is None:
                continue

            visible, ratio = is_bbox_visible(
                actor=actor,
                camera=camera,
                depth_image=depth,
                bbox=bbox,
            )

            if not visible:
                continue

            class_name = get_actor_class(actor)
            class_id = class_to_id.get(class_name)

            if class_id is None:
                continue  # unknown class, skip rather than crash

            x_min, y_min, x_max, y_max = bbox

            box_w = x_max - x_min
            box_h = y_max - y_min
            cx = x_min + box_w / 2.0
            cy = y_min + box_h / 2.0

            # Normalize to [0, 1] as YOLO expects
            norm_cx = cx / width
            norm_cy = cy / height
            norm_w = box_w / width
            norm_h = box_h / height

            label_lines.append(
                f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        # ---- Save image ----
        image_path = os.path.join(images_dir, f"frame_{current_frame:06d}.png")
        success = cv2.imwrite(image_path, rgb)

        if not success:
            raise RuntimeError(f"Failed to save image: {image_path}")

        # ---- Save labels (even if empty, YOLO expects a matching file) ----
        label_path = os.path.join(labels_dir, f"frame_{current_frame:06d}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        saved_count += 1

        print(
            f"Saved {saved_count}/{num_images} "
            f"(frame {current_frame}, {len(label_lines)} objects)"
        )

    print(f"Done. Attempted {attempted_frames} ticks, saved {saved_count} labeled frames.")


def cleanup(
    world,
    camera,
    depth_camera,
    ego_vehicle,
    vehicles,
    controllers,
    walkers,
    traffic_manager,
):
    print("Stopping...")

    if camera is not None and camera.is_alive:
        camera.stop()
        for _ in range(3):
            world.tick()
        camera.destroy()

    if depth_camera is not None and depth_camera.is_alive:
        depth_camera.stop()
        for _ in range(3):
            world.tick()
        depth_camera.destroy()

    for controller in controllers:
        if controller.is_alive:
            controller.stop()
            controller.destroy()

    for walker in walkers:
        if walker.is_alive:
            walker.destroy()

    for vehicle in vehicles:
        if vehicle.is_alive:
            vehicle.set_autopilot(False, traffic_manager.get_port())
            vehicle.destroy()

    if ego_vehicle is not None and ego_vehicle.is_alive:
        ego_vehicle.set_autopilot(False, traffic_manager.get_port())
        ego_vehicle.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    cv2.destroyAllWindows()
    print("Done")


def main():
    camera = None
    depth_camera = None
    ego_vehicle = None

    vehicles = []
    controllers = []
    walkers = []

    simulation_config = load_config("../configs/simulation.yaml")
    simulation = simulation_config["simulation"]

    traffic_config = load_config("../configs/traffic.yaml")
    traffic = traffic_config["traffic"]

    camera_config = load_config("../configs/camera.yaml")
    camera_settings = camera_config["camera"]

    client, world = connect(
        simulation["host"], simulation["port"], simulation["town"]
    )

    blueprints = world.get_blueprint_library()
    vehicle_blueprints = blueprints.filter("*vehicle*")
    walker_blueprints = blueprints.filter("*walker*")
    spawn_points = world.get_map().get_spawn_points()

    traffic_manager = setup_traffic_manager(
        client, world,
        traffic["distance_to_leading_vehicle"],
        traffic["speed_diff_percent"]
    )

    try:
        ego_vehicle = spawn_ego_vehicle(
            world, blueprints, spawn_points, traffic_manager,
            traffic["speed_diff_percent"], traffic["distance_to_leading_vehicle"]
        )

        vehicles = spawn_random_traffic(
            world, vehicle_blueprints, spawn_points, traffic_manager,
            traffic["num_vehicles"], exclude_spawn_point=spawn_points[0],
        )

        walkers, controllers = spawn_pedestrians(
            world, walker_blueprints, traffic["num_pedestrians"]
        )

        # Tick before touching controllers — pedestrian AI controllers
        # must be server-resolved before .start()/.go_to_location() work.
        world.tick()

        for controller in controllers:
            if not controller.is_alive:
                continue
            try:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(1.4 + random.uniform(-0.3, 0.3))
            except RuntimeError as e:
                print(f"Failed to start controller {controller.id}: {e}")

        print(f"Spawned {len(walkers)} pedestrians")

        # Single source of truth for the relative sensor transform.
        # RGB and depth cameras MUST share this exact object — passing
        # camera.get_transform() into the depth spawn was the earlier
        # bug (stale/unresolved world transform read before a tick,
        # and a world-space value being reused as a relative offset).
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        camera = spawn_camera(
            world, blueprints, ego_vehicle,
            camera_settings["width"], camera_settings["height"], camera_settings["fov"],
            transform=camera_transform
        )

        depth_camera = spawn_depth_camera(
            world, blueprints, ego_vehicle,
            camera_settings["width"], camera_settings["height"], camera_settings["fov"],
            transform=camera_transform
        )

        K = build_camera_matrix(
            camera_settings["width"], camera_settings["height"], camera_settings["fov"]
        )

        frame_buffer = FrameBuffer()
        depth_buffer = DepthFrameBuffer()

        camera.listen(frame_buffer.on_image)
        depth_camera.listen(depth_buffer.on_image)

        synced_frame = wait_for_synced_frame(world, frame_buffer, depth_buffer)
        print(f"Initial sync at frame: {synced_frame}")

        """
        run_perception_loop(
            world, ego_vehicle, camera, depth_camera,
            frame_buffer, depth_buffer, K
        )"""

        save_dataset(
            world, ego_vehicle, camera, depth_camera,
            frame_buffer, depth_buffer, K,
            output_dir="../data/raw/continuous_v1",
            num_images=2000,
            save_interval=20,
        )

    finally:
        cleanup(
            world, camera, depth_camera, ego_vehicle,
            vehicles, controllers, walkers, traffic_manager
        )


if __name__ == "__main__":
    main()