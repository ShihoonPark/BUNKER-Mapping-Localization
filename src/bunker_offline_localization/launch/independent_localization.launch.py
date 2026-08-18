from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("bunker_offline_localization"))
    localization_config = str(share / "config" / "localization.yaml")
    independent_config = str(share / "config" / "independent_163346.yaml")
    ekf_config = str(share / "config" / "ekf.yaml")

    bag_path = LaunchConfiguration("bag_path")
    map_path = LaunchConfiguration("map_path")
    results_directory = LaunchConfiguration("results_directory")
    playback_rate = LaunchConfiguration("playback_rate")
    max_scans = LaunchConfiguration("max_scans")
    window_origin = LaunchConfiguration("window_origin_timestamp")
    window_start = LaunchConfiguration("window_start_sec")
    window_end = LaunchConfiguration("window_end_sec")
    publish_test_identity = LaunchConfiguration("publish_test_identity_base_to_lidar")
    window_parameters = {
        "time_window.enabled": True,
        "time_window.origin_timestamp": window_origin,
        "time_window.start_offset_sec": window_start,
        "time_window.end_offset_sec": window_end,
    }

    adapter = Node(
        package="bunker_offline_localization",
        executable="covariance_adapter_node",
        name="covariance_adapter",
        parameters=[localization_config, independent_config, window_parameters],
        output="screen",
    )
    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        remappings=[("odometry/filtered", "/localization/odometry/filtered")],
        parameters=[ekf_config],
        output="screen",
    )
    localizer = Node(
        package="bunker_offline_localization",
        executable="offline_localizer_node",
        name="offline_localizer",
        parameters=[
            localization_config,
            independent_config,
            window_parameters,
            {
                "map_path": map_path,
                "results_directory": results_directory,
                "max_scans": max_scans,
            },
        ],
        output="screen",
    )

    # The actual base_link->velodyne calibration is still unavailable. Keep the same visible,
    # explicit Phase 1 approximation as the validated 150626 smoke test.
    base_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="phase1_identity_base_to_lidar",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "base_link", "--child-frame-id", "velodyne",
        ],
        condition=IfCondition(publish_test_identity),
        output="screen",
    )
    lidar_to_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="known_lidar_to_imu",
        arguments=[
            "--x", "0", "--y", "0", "--z", "-0.07",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "velodyne", "--child-frame-id", "imu_link",
        ],
        output="screen",
    )
    bag_player = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", bag_path,
            "--clock", "100.0",
            "--rate", playback_rate,
            "--delay", "3.0",
            "--disable-keyboard-controls",
            "--topics", "/odom", "/imu/data", "/velodyne_points",
        ],
        output="screen",
    )

    shutdown_after_bag = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_player,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[
                        LogInfo(msg="Bag playback ended; flushing independent-run outputs."),
                        EmitEvent(event=Shutdown(reason="offline bag completed")),
                    ],
                )
            ],
        )
    )
    shutdown_if_localizer_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=localizer,
            on_exit=[EmitEvent(event=Shutdown(reason="time-window localizer exited"))],
        )
    )

    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
        DeclareLaunchArgument(
            "bag_path",
            default_value="/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346",
        ),
        DeclareLaunchArgument(
            "map_path",
            default_value="/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply",
        ),
        DeclareLaunchArgument(
            "results_directory",
            default_value="/home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s",
        ),
        DeclareLaunchArgument("window_origin_timestamp", default_value="1786692827.2210245"),
        DeclareLaunchArgument("window_start_sec", default_value="0.0"),
        DeclareLaunchArgument("window_end_sec", default_value="50.0"),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("max_scans", default_value="0"),
        DeclareLaunchArgument("publish_test_identity_base_to_lidar", default_value="true"),
        adapter,
        ekf,
        localizer,
        base_to_lidar,
        lidar_to_imu,
        shutdown_after_bag,
        shutdown_if_localizer_exits,
        bag_player,
    ])
