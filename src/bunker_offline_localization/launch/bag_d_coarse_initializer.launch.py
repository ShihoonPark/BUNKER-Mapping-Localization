from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag_path = LaunchConfiguration("bag_path")
    map_path = LaunchConfiguration("map_path")
    output_directory = LaunchConfiguration("output_directory")

    initializer = Node(
        package="bunker_offline_localization",
        executable="coarse_global_initializer_node",
        name="coarse_global_initializer",
        parameters=[
            {
                "map_path": map_path,
                "output_directory": output_directory,
                "use_sim_time": True,
            }
        ],
        output="screen",
    )
    bag_player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--clock",
            "100.0",
            "--rate",
            "1.0",
            "--delay",
            "3.0",
            "--disable-keyboard-controls",
            "--topics",
            "/velodyne_points",
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
            DeclareLaunchArgument(
                "bag_path",
                default_value=(
                    "/home/a/Desktop/shihoon/Slam/20260819_dataset/"
                    "bag_D_flat_independent_localization"
                ),
            ),
            DeclareLaunchArgument(
                "map_path",
                default_value=(
                    "/home/a/Desktop/shihoon/glim_real/20260819_flat/results/"
                    "flat_bag_C_imu_on.ply"
                ),
            ),
            DeclareLaunchArgument(
                "output_directory",
                default_value=(
                    "/home/a/Desktop/shihoon/bunker_localization_ws/results/"
                    "bag_D_flat_20260819_coarse_init"
                ),
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=initializer,
                    on_exit=[EmitEvent(event=Shutdown(reason="coarse initializer completed"))],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[EmitEvent(event=Shutdown(reason="bag ended before initialization"))],
                )
            ),
            initializer,
            bag_player,
        ]
    )
