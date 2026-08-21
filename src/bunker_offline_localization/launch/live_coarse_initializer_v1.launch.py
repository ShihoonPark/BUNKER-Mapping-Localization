from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_MAP = (
    "/home/a/Desktop/shihoon/glim_real/20260819_flat/results/"
    "flat_bag_C_imu_on.ply"
)
DEFAULT_OUTPUT = (
    "/home/a/Desktop/shihoon/bunker_localization_ws/results/"
    "live_real_localization_v1/coarse"
)


def generate_launch_description():
    share = Path(get_package_share_directory("bunker_offline_localization"))
    live_config = str(share / "config" / "live_real_localization_v1.yaml")
    initializer = Node(
        package="bunker_offline_localization",
        executable="coarse_global_initializer_node",
        name="coarse_global_initializer",
        parameters=[
            live_config,
            {
                "map_path": LaunchConfiguration("map_path"),
                "output_directory": LaunchConfiguration("output_directory"),
            },
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
            DeclareLaunchArgument("map_path", default_value=DEFAULT_MAP),
            DeclareLaunchArgument("output_directory", default_value=DEFAULT_OUTPUT),
            LogInfo(
                msg=(
                    "Live coarse initialization: keep the robot stationary; consumes one "
                    "usable /velodyne_points scan and never publishes /cmd_vel."
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=initializer,
                    on_exit=[EmitEvent(event=Shutdown(reason="live coarse initializer completed"))],
                )
            ),
            initializer,
        ]
    )
