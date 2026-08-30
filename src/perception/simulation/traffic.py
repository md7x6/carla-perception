import carla 
import random

def setup_traffic_manager(client, world, distance_to_leading_vehicle, speed_diff_percent):
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(distance_to_leading_vehicle)
    traffic_manager.global_percentage_speed_difference(speed_diff_percent)

    # Synchronous mode keeps the server, the Traffic Manager thread, and this
    # script's main loop all stepping in lockstep. This avoids a whole class
    # of race conditions (e.g. TM acting on an actor that's mid-destruction)
    # that show up as "trying to operate on a destroyed actor" crashes.
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    return traffic_manager

def spawn_pedestrians(world, walker_blueprints, count):
    walkers = []
    controllers = []

    controller_bp = world.get_blueprint_library().find(
        "controller.ai.walker"
    )

    for _ in range(count):

        loc = world.get_random_location_from_navigation()

        if loc is None:
            continue

        walker_bp = random.choice(walker_blueprints)

        walker = world.try_spawn_actor(
            walker_bp,
            carla.Transform(location=loc)
        )

        if walker is None:
            continue

        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),
            attach_to=walker
        )

        if controller is None:
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)

    return walkers, controllers

def spawn_random_traffic(world, vehicle_blueprints, spawn_points, traffic_manager, count,
                          exclude_spawn_point=None):
    vehicles = []

    # Never spawn traffic on top of a point another actor (e.g. the ego) already used.
    available_points = [p for p in spawn_points if p != exclude_spawn_point]
    random.shuffle(available_points)
    available_points = available_points[:count]

    for spawn_point in available_points:
        blueprint = random.choice(vehicle_blueprints)

        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)

    print(f"Spawned {len(vehicles)} traffic vehicles")
    return vehicles