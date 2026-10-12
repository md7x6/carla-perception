
from builtin_interfaces.msg import Time as TimeMsg


def carla_time_to_ros(simulation_time):
    """
    Convert CARLA simulation time (seconds)
    into a ROS2 Time message.
    """
    nanoseconds = round(simulation_time * 1_000_000_000)

    stamp = TimeMsg()
    stamp.sec = nanoseconds // 1_000_000_000
    stamp.nanosec = nanoseconds % 1_000_000_000

    return stamp
