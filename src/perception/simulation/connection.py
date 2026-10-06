import carla

def connect(host, port, town):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.load_world(town)
    world.set_weather(carla.WeatherParameters.WetCloudySunset)

    print("Server: ", client.get_server_version())
    print("World:", world.get_map().name)

    return client, world