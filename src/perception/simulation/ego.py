def spawn_ego_vehicle(world, blueprints, spawn_points, traffic_manager,
                       speed_diff_percent, distance_to_leading_vehicle):
    ego_blueprint = blueprints.find("vehicle.tesla.model3")
    print("Ego vehicle blueprint:", ego_blueprint.id)

    spawn_point = spawn_points[0]
    ego_vehicle = world.try_spawn_actor(ego_blueprint, spawn_point)

    if ego_vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle at spawn_points[0]")

    print("Ego vehicle spawned:", ego_vehicle)

    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, speed_diff_percent)
    traffic_manager.distance_to_leading_vehicle(ego_vehicle, distance_to_leading_vehicle)
    traffic_manager.auto_lane_change(ego_vehicle, False)

    return ego_vehicle