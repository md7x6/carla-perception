import carla
import numpy as np


def carla_depth_to_meters(image):
    """
    Convert CARLA depth image to meters.

    CARLA raw image format is BGRA, while the depth encoding
    is defined using RGB:

        normalized =
            (R + G * 256 + B * 256^2) / (256^3 - 1)

        depth_meters = normalized * 1000
    """

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    ).reshape((image.height, image.width, 4))

    # IMPORTANT:
    # raw_data is BGRA
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)

    normalized = (
        r
        + g * 256.0
        + b * 65536.0
    ) / 16777215.0

    return normalized * 1000.0


def get_actor_depth_range(actor, camera):
    """
    Return the near and far depth of the actor's 3D bounding box
    along the camera forward axis.
    """

    camera_transform = camera.get_transform()

    camera_location = camera_transform.location
    camera_forward = camera_transform.get_forward_vector()

    corners = actor.bounding_box.get_world_vertices(
        actor.get_transform()
    )

    depths = []

    for corner in corners:
        direction = corner - camera_location

        depth = (
            direction.x * camera_forward.x
            + direction.y * camera_forward.y
            + direction.z * camera_forward.z
        )

        depths.append(depth)

    return min(depths), max(depths)


def visibility_ratio(
    actor,
    camera,
    depth_image,
    bbox,
    depth_tolerance=0.5,
    sample_step=5,
    debug=False,
):
    """
    Estimate how much of the projected bbox is visible.

    A pixel is considered visible when the observed depth is
    close to the actor's nearest surface depth.

    This is a first practical depth-based visibility test.
    """

    if bbox is None:
        return 0.0

    x_min, y_min, x_max, y_max = bbox

    height, width = depth_image.shape

    # Clip bbox to image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)

    if x_min >= x_max or y_min >= y_max:
        return 0.0

    near_depth, far_depth = get_actor_depth_range(
        actor,
        camera
    )

    if far_depth <= 0:
        return 0.0

    # Only use actors whose front surface is in front of camera.
    near_depth = max(0.0, near_depth)

    # FIX 1: scale tolerance with distance instead of a flat value.
    # CARLA's 24-bit depth encoding over a 1000m range loses precision
    # at longer distances, so a fixed tolerance that works up close
    # becomes too tight for far-away, genuinely-visible actors.
    tolerance = max(depth_tolerance, 0.02 * far_depth)

    near_bound = near_depth - tolerance
    far_bound = far_depth + tolerance

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    center_depth = depth_image[cy, cx]

    if debug:
        print("=" * 70)
        print(f"Actor: {actor.id} {actor.type_id}")
        print(f"BBox: ({x_min}, {y_min}, {x_max}, {y_max})")
        print(f"BBox center: ({cx}, {cy})")
        print(f"Actor depth: {near_depth:.3f} -> {far_depth:.3f} m")
        print(f"Tolerance used: {tolerance:.3f} m")
        print(f"Depth at bbox center: {center_depth:.3f} m")
        print(
            f"Center inside range: "
            f"{near_bound <= center_depth <= far_bound}"
        )
        print("=" * 70)

    # FIX 2: build the sample point set so small bboxes always get at
    # least a few samples, instead of relying purely on a fixed step
    # that can skip over tiny (e.g. 4x10 px) boxes almost entirely.
    sample_points = set()

    for y in range(y_min, y_max + 1, sample_step):
        for x in range(x_min, x_max + 1, sample_step):
            sample_points.add((x, y))

    # Always include center and all four corners regardless of step.
    sample_points.add((cx, cy))
    sample_points.add((x_min, y_min))
    sample_points.add((x_max, y_min))
    sample_points.add((x_min, y_max))
    sample_points.add((x_max, y_max))

    valid_pixels = 0
    visible_pixels = 0

    for x, y in sample_points:

        observed_depth = depth_image[y, x]

        if not np.isfinite(observed_depth):
            continue

        if observed_depth <= 0:
            continue

        valid_pixels += 1

        # An object in front of our actor will have a
        # significantly smaller depth.
        #
        # We accept depths from the actor's near surface
        # to its far surface, expanded by a distance-scaled tolerance.
        if near_bound <= observed_depth <= far_bound:
            visible_pixels += 1

    if valid_pixels == 0:
        return 0.0

    return visible_pixels / valid_pixels


def is_bbox_visible(
    actor,
    camera,
    depth_image,
    bbox,
    min_visible_ratio=0.20,
    min_pixel_area=20,
    depth_tolerance=0.5,
    sample_step=5,
):
    """
    Determine whether enough of an actor is visible.

    Returns:
        (visible, ratio) tuple.
        visible: True if the actor passes the visibility checks.
        ratio: the computed visibility ratio (0.0 if rejected early).
    """

    if bbox is None:
        return False, 0.0

    x_min, y_min, x_max, y_max = bbox

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return False, 0.0

    area = width * height

    if area < min_pixel_area:
        return False, 0.0

    ratio = visibility_ratio(
        actor=actor,
        camera=camera,
        depth_image=depth_image,
        bbox=bbox,
        depth_tolerance=depth_tolerance,
        sample_step=sample_step,
    )

    return ratio >= min_visible_ratio, ratio