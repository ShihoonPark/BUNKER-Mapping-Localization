import json
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


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _validate_mode(mode):
    if mode not in {"short", "full"}:
        raise RuntimeError("mode must be short or full")


def _window_parameters(mode):
    _validate_mode(mode)
    return {
        "time_window.enabled": mode != "full",
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


def _bag_command(mode, bag_path):
    _validate_mode(mode)
    command = [
        "ros2", "bag", "play", bag_path,
        "--clock", "100.0", "--rate", "1.0", "--delay", "3.0",
        "--disable-keyboard-controls",
    ]
    if mode == "full":
        command.extend(["--wait-for-all-acked", "5000"])
    command.extend(["--topics", "/odom", "/imu/data", "/velodyne_points"])
    return command


def _setup(context):
    mode = _value(context, "mode")
    _validate_mode(mode)

    share = Path(get_package_share_directory("bunker_offline_localization"))
    localization_config = str(share / "config" / "localization.yaml")
    independent_config = str(share / "config" / "independent_bag_D_20260819.yaml")
    ekf_config = str(share / "config" / "ekf.yaml")
    full = mode == "full"
    window_parameters = _window_parameters(mode)
    output_directory = _value(
        context, "full_results_directory" if full else "short_results_directory"
    )
    seed_parameters = _seed_parameters(_value(context, "seed_json"))
    full_parameters = {
        "full_bag.enabled": full,
        "full_bag.timing_gap_threshold_sec": 1.25,
        "full_bag.replay_rate": 1.0,
        "full_bag.expected_lidar_inputs": BAG_LIDAR_COUNT,
        "full_bag.eof_topic": "/localization/bag_d_full_eof",
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
            seed_parameters,
            full_parameters,
            {
                "map_path": _value(context, "map_path"),
                "results_directory": output_directory,
                "max_scans": 0,
            },
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
        output="screen",
    )
    lidar_to_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="phase1_lidar_to_imu_approximation",
        arguments=[
            "--x", "0", "--y", "0", "--z", "-0.07",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "velodyne", "--child-frame-id", "imu_link",
        ],
        output="screen",
    )
    bag_command = _bag_command(mode, _value(context, "bag_path"))
    bag_player = ExecuteProcess(cmd=bag_command, output="screen")
    actions = [adapter, ekf, localizer, base_to_lidar, lidar_to_imu]
    if full:
        eof_notifier = ExecuteProcess(
            cmd=[
                "ros2", "topic", "pub", "--once",
                "/localization/bag_d_full_eof", "std_msgs/msg/Empty", "{}",
            ],
            output="screen",
        )

        def on_bag_exit(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason="Bag D playback failed"))]
            return [
                TimerAction(
                    period=3.0,
                    actions=[
                        LogInfo(msg="Bag D reached EOF; finalizing full localization."),
                        eof_notifier,
                        TimerAction(
                            period=2.0,
                            actions=[EmitEvent(event=Shutdown(reason="full Bag D completed"))],
                        ),
                    ],
                )
            ]

        actions.append(
            RegisterEventHandler(OnProcessExit(target_action=bag_player, on_exit=on_bag_exit))
        )
    else:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=localizer,
                    on_exit=[EmitEvent(event=Shutdown(reason="Bag D short Gate completed"))],
                )
            )
        )
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[EmitEvent(event=Shutdown(reason="Bag D ended unexpectedly"))],
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
            DeclareLaunchArgument("mode", default_value="short"),
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
                "short_results_directory",
                default_value=f"{root}/results/bag_D_flat_20260819_localization_0_50s",
            ),
            DeclareLaunchArgument(
                "full_results_directory",
                default_value=f"{root}/results/bag_D_flat_20260819_localization_full",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
