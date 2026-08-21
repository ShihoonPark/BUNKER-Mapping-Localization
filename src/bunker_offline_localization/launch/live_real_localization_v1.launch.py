import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_MAP = (
    "/home/a/Desktop/shihoon/glim_real/20260819_flat/results/"
    "flat_bag_C_imu_on.ply"
)
DEFAULT_SEED = (
    "/home/a/Desktop/shihoon/bunker_localization_ws/results/"
    "live_real_localization_v1/coarse/coarse_initialization.json"
)
DEFAULT_RESULTS = (
    "/home/a/Desktop/shihoon/bunker_localization_ws/results/"
    "live_real_localization_v1/continuous"
)


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _seed_parameters(seed_path, expected_map_path=None):
    path = Path(seed_path)
    seed = json.loads(path.read_text(encoding="utf-8"))
    if not seed.get("success") or not seed.get("best_candidate"):
        raise RuntimeError(f"coarse initializer did not produce a valid seed: {path}")
    if seed.get("transform_convention") != "p_map = T_map_lidar * p_lidar":
        raise RuntimeError("coarse seed has an unexpected transform convention")
    if seed.get("scans_used") != 1:
        raise RuntimeError("live coarse seed must use exactly one scan")
    if expected_map_path is not None:
        recorded_map = Path(seed.get("map_path", "")).resolve()
        if recorded_map != Path(expected_map_path).resolve():
            raise RuntimeError(
                f"coarse seed map mismatch: seed={recorded_map} launch={expected_map_path}"
            )
    pose = seed["best_candidate"]["T_map_lidar"]
    translation = [float(value) for value in pose["translation"]]
    rotation = [float(value) for value in pose["rotation_xyzw"]]
    if len(translation) != 3 or len(rotation) != 4:
        raise RuntimeError("coarse seed has invalid transform dimensions")
    if not all(math.isfinite(value) for value in translation + rotation):
        raise RuntimeError("coarse seed transform contains non-finite values")
    quaternion_norm = math.sqrt(sum(value * value for value in rotation))
    if abs(quaternion_norm - 1.0) > 1.0e-3:
        raise RuntimeError("coarse seed quaternion is not normalized")
    return {
        "initialization.mode": "parameter",
        "initialization.translation": translation,
        "initialization.rotation_xyzw": rotation,
    }


def _setup(context):
    share = Path(get_package_share_directory("bunker_offline_localization"))
    localization_config = str(share / "config" / "localization.yaml")
    ekf_config = str(share / "config" / "ekf.yaml")
    live_config = str(share / "config" / "live_real_localization_v1.yaml")
    map_path = _value(context, "map_path")
    seed_parameters = _seed_parameters(_value(context, "seed_json"), map_path)

    adapter = Node(
        package="bunker_offline_localization",
        executable="covariance_adapter_node",
        name="covariance_adapter",
        parameters=[localization_config, live_config],
        output="screen",
    )
    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        remappings=[("odometry/filtered", "/localization/odometry/filtered")],
        parameters=[ekf_config, live_config],
        output="screen",
    )
    localizer = Node(
        package="bunker_offline_localization",
        executable="offline_localizer_node",
        name="offline_localizer",
        parameters=[
            localization_config,
            live_config,
            seed_parameters,
            {
                "map_path": map_path,
                "results_directory": _value(context, "results_directory"),
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
    return [
        LogInfo(
            msg=(
                "PHASE-1 PLANAR APPROXIMATION / NOT CALIBRATED: "
                "T_base_lidar is identity; no /cmd_vel publisher is launched."
            )
        ),
        adapter,
        ekf,
        localizer,
        base_to_lidar,
        lidar_to_imu,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
            DeclareLaunchArgument("map_path", default_value=DEFAULT_MAP),
            DeclareLaunchArgument("seed_json", default_value=DEFAULT_SEED),
            DeclareLaunchArgument("results_directory", default_value=DEFAULT_RESULTS),
            OpaqueFunction(function=_setup),
        ]
    )
