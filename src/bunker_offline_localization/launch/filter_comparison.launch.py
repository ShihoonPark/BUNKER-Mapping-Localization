import signal
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, share):
    filter_type = LaunchConfiguration("filter_type").perform(context).lower()
    dataset = LaunchConfiguration("dataset").perform(context).lower()
    if filter_type not in {"ekf", "ukf"}:
        raise RuntimeError("filter_type must be 'ekf' or 'ukf'")
    if dataset not in {"same_bag", "independent"}:
        raise RuntimeError("dataset must be 'same_bag' or 'independent'")

    localization_config = str(share / "config" / "localization.yaml")
    independent_config = str(share / "config" / "independent_163346.yaml")
    filter_common = str(share / "config" / "filter_common.yaml")
    filter_specific = str(share / "config" / f"filter_{filter_type}.yaml")
    results_directory = LaunchConfiguration("results_directory")
    max_scans = LaunchConfiguration("max_scans")
    publish_test_identity = LaunchConfiguration("publish_test_identity_base_to_lidar")

    if dataset == "same_bag":
        bag_path = "/home/a/Desktop/shihoon/Slam/slam_20260814_150626"
        localizer_configs = [localization_config]
        adapter_configs = [localization_config]
    else:
        bag_path = "/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346"
        localizer_configs = [localization_config, independent_config]
        adapter_configs = [localization_config, independent_config]

    adapter = Node(
        package="bunker_offline_localization",
        executable="covariance_adapter_node",
        name="covariance_adapter",
        parameters=adapter_configs,
        output="screen",
    )
    localization_filter = Node(
        package="robot_localization",
        executable=f"{filter_type}_node",
        name="localization_filter_node",
        remappings=[("odometry/filtered", "/localization/odometry/filtered")],
        parameters=[filter_common, filter_specific],
        output="screen",
    )
    timing_monitor = Node(
        package="bunker_offline_localization",
        executable="filter_timing_monitor_node",
        name="filter_timing_monitor",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    localizer = Node(
        package="bunker_offline_localization",
        executable="offline_localizer_node",
        name="offline_localizer",
        parameters=localizer_configs
        + [
            {
                "results_directory": results_directory,
                "max_scans": max_scans,
                "filter_type_label": filter_type,
                "require_filter_timing": True,
            }
        ],
        output="screen",
    )
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
            "--rate", "1.0",
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
                        LogInfo(msg=f"{dataset}/{filter_type} playback ended; flushing outputs."),
                        EmitEvent(event=Shutdown(reason="comparison playback completed")),
                    ],
                )
            ],
        )
    )
    shutdown_if_localizer_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=localizer,
            on_exit=[EmitEvent(event=Shutdown(reason="comparison localizer exited"))],
        )
    )

    actions = [
        adapter,
        localization_filter,
        timing_monitor,
        localizer,
        base_to_lidar,
        lidar_to_imu,
        shutdown_after_bag,
        shutdown_if_localizer_exits,
        bag_player,
    ]
    if dataset == "independent":
        # Humble ros2 bag play has no duration option. The data path itself enforces [0, 50] s;
        # this wall timer only prevents replaying the known later gap after a 1x run.
        actions.append(
            TimerAction(
                period=54.0,
                actions=[
                    EmitEvent(
                        event=SignalProcess(
                            signal_number=signal.SIGINT,
                            process_matcher=lambda action: action is bag_player,
                        )
                    )
                ],
            )
        )
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory("bunker_offline_localization"))
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
            DeclareLaunchArgument("filter_type", default_value="ekf"),
            DeclareLaunchArgument("dataset", default_value="same_bag"),
            DeclareLaunchArgument(
                "results_directory",
                default_value="/home/a/Desktop/shihoon/bunker_localization_ws/results/filter_comparison/manual",
            ),
            DeclareLaunchArgument("max_scans", default_value="0"),
            DeclareLaunchArgument("publish_test_identity_base_to_lidar", default_value="true"),
            OpaqueFunction(function=_launch_setup, args=[share]),
        ]
    )
