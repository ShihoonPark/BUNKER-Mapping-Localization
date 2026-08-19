import json
import math
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
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BAG_ORIGIN_TIMESTAMP = 1787142248.3215761
BAG_LIDAR_COUNT = 2181
INDEPENDENT_CONFIG = "independent_bag_D_20260819.yaml"
EKF_CONFIG = "ekf.yaml"
DIAGNOSTIC_CONFIG = "gicp_rviz_diagnostic.yaml"
BASE_TO_LIDAR_ARGUMENTS = [
    "--x", "0", "--y", "0", "--z", "0",
    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
    "--frame-id", "base_link", "--child-frame-id", "velodyne",
]
LIDAR_TO_IMU_ARGUMENTS = [
    "--x", "0", "--y", "0", "--z", "-0.07",
    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
    "--frame-id", "velodyne", "--child-frame-id", "imu_link",
]


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _validate_arguments(playback_rate, max_scans):
    if not math.isfinite(playback_rate) or playback_rate <= 0.0:
        raise RuntimeError("playback_rate must be finite and positive")
    if max_scans < 0:
        raise RuntimeError("max_scans must be nonnegative")


def _production_verification_mode(playback_rate, max_scans):
    _validate_arguments(playback_rate, max_scans)
    return abs(playback_rate - 1.0) <= 1.0e-12 and max_scans == 0


def _window_parameters():
    return {
        "time_window.enabled": False,
        "time_window.origin_timestamp": BAG_ORIGIN_TIMESTAMP,
        "time_window.start_offset_sec": 0.0,
        "time_window.end_offset_sec": 50.0,
    }


def _seed_parameters(seed_path):
    seed_path = Path(seed_path)
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not seed.get("success") or not seed.get("best_candidate"):
        raise RuntimeError(f"coarse initializer did not produce a valid seed: {seed_path}")
    pose = seed["best_candidate"]["T_map_lidar"]
    translation = [float(value) for value in pose["translation"]]
    rotation = [float(value) for value in pose["rotation_xyzw"]]
    if len(translation) != 3 or len(rotation) != 4:
        raise RuntimeError("coarse seed has invalid transform dimensions")
    return {
        "initialization.mode": "parameter",
        "initialization.translation": translation,
        "initialization.rotation_xyzw": rotation,
    }


def _full_bag_parameters(playback_rate, max_scans):
    production_verification = _production_verification_mode(playback_rate, max_scans)
    return {
        "full_bag.enabled": production_verification,
        "full_bag.timing_gap_threshold_sec": 1.25,
        "full_bag.replay_rate": 1.0 if production_verification else playback_rate,
        "full_bag.expected_lidar_inputs": BAG_LIDAR_COUNT,
        "full_bag.eof_topic": "/localization/bag_d_rviz_eof",
    }


def _bag_command(bag_path, playback_rate, complete_bag):
    command = [
        "ros2", "bag", "play", bag_path,
        "--clock", "100.0", "--rate", str(playback_rate), "--delay", "3.0",
        "--disable-keyboard-controls",
    ]
    if complete_bag:
        command.extend(["--wait-for-all-acked", "5000"])
    command.extend(["--topics", "/odom", "/imu/data", "/velodyne_points"])
    return command


def _setup(context):
    playback_rate = float(_value(context, "playback_rate"))
    max_scans = int(_value(context, "max_scans"))
    production_verification = _production_verification_mode(playback_rate, max_scans)
    complete_bag = max_scans == 0
    share = Path(get_package_share_directory("bunker_offline_localization"))
    localization_config = str(share / "config" / "localization.yaml")
    independent_config = str(share / "config" / INDEPENDENT_CONFIG)
    diagnostic_config = str(share / "config" / DIAGNOSTIC_CONFIG)
    ekf_config = str(share / "config" / EKF_CONFIG)
    results_directory = _value(context, "results_directory")

    window_parameters = _window_parameters()
    seed_parameters = _seed_parameters(_value(context, "seed_json"))
    full_bag_parameters = _full_bag_parameters(playback_rate, max_scans)
    diagnostic_parameters = {
        "diagnostics.enabled": True,
        "diagnostics.publish_visualization": True,
        "diagnostics.publish_correspondences_every_n_scans": int(
            _value(context, "publish_correspondences_every_n_scans")
        ),
        "diagnostics.rviz_max_correspondence_lines": int(
            _value(context, "rviz_max_correspondence_lines")
        ),
        "diagnostics.output_directory": results_directory,
        "diagnostics.hold_selected_role": "",
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
            diagnostic_config,
            window_parameters,
            seed_parameters,
            full_bag_parameters,
            diagnostic_parameters,
            {
                "map_path": _value(context, "map_path"),
                "results_directory": results_directory,
                "max_scans": max_scans,
            },
        ],
        output="screen",
    )
    base_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="phase1_identity_base_to_lidar",
        arguments=BASE_TO_LIDAR_ARGUMENTS,
        output="screen",
    )
    lidar_to_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="phase1_lidar_to_imu_approximation",
        arguments=LIDAR_TO_IMU_ARGUMENTS,
        output="screen",
    )
    bag_player = ExecuteProcess(
        cmd=_bag_command(
            _value(context, "bag_path"), playback_rate, complete_bag
        ),
        output="screen",
    )

    actions = [adapter, ekf, localizer, base_to_lidar, lidar_to_imu]
    if _value(context, "launch_rviz").lower() == "true":
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="bag_d_localization_rviz",
                arguments=["-d", str(share / "rviz" / "gicp_live_diagnostic.rviz")],
                output="screen",
            )
        )

    eof_notifier = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            "/localization/bag_d_rviz_eof", "std_msgs/msg/Empty", "{}",
        ],
        output="screen",
    )

    def on_bag_exit(event, _context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(reason="Bag D playback failed"))]
        if production_verification:
            return [
                TimerAction(
                    period=3.0,
                    actions=[
                        LogInfo(
                            msg=(
                                "Bag D 1x reached EOF; finalizing accounting and "
                                "keeping the final RViz state until Ctrl+C."
                            )
                        ),
                        eof_notifier,
                    ],
                )
            ]
        return [
            LogInfo(
                msg=(
                    "Bag D visualization playback reached EOF; keeping map, registered "
                    "scan, poses, and path visible until Ctrl+C."
                )
            )
        ]

    actions.append(
        RegisterEventHandler(OnProcessExit(target_action=bag_player, on_exit=on_bag_exit))
    )
    actions.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=localizer,
                on_exit=[EmitEvent(event=Shutdown(reason="Bag D localizer exited"))],
            )
        )
    )
    actions.append(bag_player)
    return actions


def generate_launch_description():
    root = "/home/a/Desktop/shihoon/bunker_localization_ws"
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument("max_scans", default_value="0"),
            DeclareLaunchArgument(
                "publish_correspondences_every_n_scans", default_value="1"
            ),
            DeclareLaunchArgument("rviz_max_correspondence_lines", default_value="200"),
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
                "seed_json",
                default_value=(
                    f"{root}/results/bag_D_flat_20260819_coarse_init/"
                    "coarse_initialization.json"
                ),
            ),
            DeclareLaunchArgument(
                "results_directory",
                default_value=f"{root}/results/bag_D_rviz_localization",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
