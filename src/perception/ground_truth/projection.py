import math
import carla
import numpy as np


# Points closer than this (in meters, along camera forward axis)
# are treated as behind the camera. Using 0 here lets points very
# near the camera plane explode to huge pixel values, which can
# distort a bbox even when it's clipped afterward.
NEAR_CLIP = 0.01


def build_camera_matrix(width, height, fov):
    """
    Build the camera intrinsic matrix K.
    """

    focal_length = width / (
        2.0 * np.tan(np.deg2rad(fov) / 2.0)
    )

    return np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)


def get_bbox_corners(actor):
    """
    Get the 8 bounding-box corners in world coordinates.
    """

    bbox = actor.bounding_box
    extent = bbox.extent

    corners = [
        carla.Vector3D(
            x=sx * extent.x,
            y=sy * extent.y,
            z=sz * extent.z
        )
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]

    # Bounding box transform relative to actor
    bbox_transform = carla.Transform(
        bbox.location,
        bbox.rotation
    )

    # Actor transform in world
    actor_transform = actor.get_transform()

    world_corners = []

    for corner in corners:
        # Local bbox -> actor coordinates
        point = bbox_transform.transform(corner)

        # Actor coordinates -> world coordinates
        point = actor_transform.transform(point)

        world_corners.append(point)

    return world_corners


def world_to_camera(point, camera):
    """
    Transform a world-space point into camera coordinates.
    """

    matrix = np.array(
        camera.get_transform().get_inverse_matrix()
    )

    point_world = np.array([
        point.x,
        point.y,
        point.z,
        1.0
    ])

    return matrix @ point_world


def project_point(point_camera, K):
    """
    Project a camera-space point onto the image plane.

    CARLA camera coordinates:
        X = forward
        Y = right
        Z = up
    """

    x = point_camera[1]
    y = -point_camera[2]
    z = point_camera[0]

    # Behind camera (or too close, which blows up the projection)
    if z <= NEAR_CLIP:
        return None

    pixel = K @ np.array([
        x,
        y,
        z
    ])

    u = pixel[0] / pixel[2]
    v = pixel[1] / pixel[2]

    return float(u), float(v)


def project_bbox(actor, camera, K):
    """
    Project an actor's 3D bounding box into image coordinates.

    Returns:
        List of projected points.
        Returns [] if all bounding-box points are behind
        the camera.
    """

    world_corners = get_bbox_corners(actor)

    pixels = []

    for corner in world_corners:

        camera_point = world_to_camera(
            corner,
            camera
        )

        pixel = project_point(
            camera_point,
            K
        )

        if pixel is not None:
            pixels.append(pixel)

    return pixels


def bbox_2d_from_points(points, width, height):
    """
    Create a clipped 2D bounding box from projected points.

    Returns:
        (x_min, y_min, x_max, y_max)

    Returns None if the projected bounding box does not
    intersect the camera image.
    """

    if not points:
        return None

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    # Completely outside image
    if x_max < 0 or x_min >= width:
        return None

    if y_max < 0 or y_min >= height:
        return None

    # Clip to image
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)

    # Reject invalid / tiny boxes
    if x_max <= x_min:
        return None

    if y_max <= y_min:
        return None

    if (x_max - x_min) < 2:
        return None

    if (y_max - y_min) < 2:
        return None

    # floor the min edge, ceil the max edge so the box isn't
    # shrunk by truncation on the max side
    return (
        int(math.floor(x_min)),
        int(math.floor(y_min)),
        int(math.ceil(x_max)),
        int(math.ceil(y_max))
    )

def debug_actor_projection(actor, camera, K):
    """
    Debug one actor's position through the projection pipeline.
    """

    actor_location = actor.get_transform().location

    camera_matrix = np.array(
        camera.get_transform().get_inverse_matrix()
    )

    print("Camera matrix: ", camera_matrix)

    world_point = np.array([
        actor_location.x,
        actor_location.y,
        actor_location.z,
        1.0
    ])

    camera_point = camera_matrix @ world_point

    pixel = project_point(camera_point, K)

    print("\n========== PROJECTION DEBUG ==========")
    print(f"Actor: {actor.id} {actor.type_id}")
    print(
        "World position:",
        f"X={actor_location.x:.2f}",
        f"Y={actor_location.y:.2f}",
        f"Z={actor_location.z:.2f}"
    )
    print(
        "Camera coordinates:",
        f"X={camera_point[0]:.2f}",
        f"Y={camera_point[1]:.2f}",
        f"Z={camera_point[2]:.2f}"
    )
    print("Projected pixel:", pixel)
    print("======================================\n")

    return pixel 