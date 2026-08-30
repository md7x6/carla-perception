import carla

def get_target_actors(world):
    """
    Return vehicles and pedestrians currently present in the world.
    """

    actors = world.get_actors()

    targets = []

    for actor in actors:

        if actor.type_id.startswith("vehicle."):
            targets.append(actor)

        elif actor.type_id.startswith("walker.pedestrian."):
            targets.append(actor)

    return targets

def get_actor_ground_truth(actor):
    """
    Extract the actor's world transform and 3D bounding box.
    """

    transform = actor.get_transform()
    bounding_box = actor.bounding_box

    return {
        "actor_id": actor.id,
        "class": get_actor_class(actor),
        "type_id": actor.type_id,

        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        },

        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },

        "bounding_box": {
            "location": {
                "x": bounding_box.location.x,
                "y": bounding_box.location.y,
                "z": bounding_box.location.z,
            },

            "extent": {
                "x": bounding_box.extent.x,
                "y": bounding_box.extent.y,
                "z": bounding_box.extent.z,
            },

            "rotation": {
                "pitch": bounding_box.rotation.pitch,
                "yaw": bounding_box.rotation.yaw,
                "roll": bounding_box.rotation.roll,
            },
        },
    }


def get_actor_class(actor):
    """
    Convert CARLA actor type into our dataset class.
    """

    if actor.type_id.startswith("vehicle."):
        return "vehicle"

    if actor.type_id.startswith("walker.pedestrian."):
        return "pedestrian"

    return "unknown"


def collect_ground_truth(world):
    """
    Collect ground-truth information for all target actors.
    """

    actors = get_target_actors(world)

    ground_truth = []

    for actor in actors:

        if not actor.is_alive:
            continue

        ground_truth.append(
            get_actor_ground_truth(actor)
        )

    return ground_truth