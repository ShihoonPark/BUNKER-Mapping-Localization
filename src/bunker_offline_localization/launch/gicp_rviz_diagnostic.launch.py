import csv
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


VALID_ROLES = {
    "baseline_representative",
    "largest_negative_dz",
    "largest_mapping_z_discrepancy",
    "recovery_representative",
    "stable_stage_representative",
}


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _setup(context):
    share = Path(get_package_share_directory("bunker_offline_localization"))
    mode = _value(context, "mode")
    selected_role = _value(context, "selected_role")
    if mode not in {"continuous", "selected"}:
        raise RuntimeError("mode must be continuous or selected")
    if mode == "selected" and selected_role not in VALID_ROLES:
        raise RuntimeError(f"unsupported selected_role: {selected_role}")

    handoff_path = Path(_value(context, "selected_scans_handoff"))
    with handoff_path.open(encoding="utf-8") as stream:
        handoff = json.load(stream)
    selected = handoff["selected_scans"]
    roles = [item["selection_role"] for item in selected]
    timestamps = [float(item["timestamp"]) for item in selected]
    if set(roles) != VALID_ROLES:
        raise RuntimeError("selected scan handoff does not contain the fixed five roles")
    with Path(_value(context, "production_localization_csv")).open(
        newline="", encoding="utf-8"
    ) as stream:
        production_rows = list(csv.DictReader(stream))
    audit_rows = [production_rows[int(item["row_index"])] for item in selected]
    for item, row in zip(selected, audit_rows):
        if abs(float(row["timestamp"]) - float(item["timestamp"])) > 1.0e-6:
            raise RuntimeError("handoff row_index does not match production localization timestamp")

    def poses(prefix):
        values = []
        for row in audit_rows:
            values.extend(float(row[f"{prefix}_{field}"]) for field in ("x", "y", "z", "qx", "qy", "qz", "qw"))
        return values

    localization_config = str(share / "config" / "localization.yaml")
    independent_config = str(share / "config" / "independent_163346.yaml")
    diagnostic_config = str(share / "config" / "gicp_rviz_diagnostic.yaml")
    ekf_config = str(share / "config" / "ekf.yaml")
    window_parameters = {
        "time_window.enabled": True,
        "time_window.origin_timestamp": 1786692827.2210245,
        "time_window.start_offset_sec": 0.0,
        "time_window.end_offset_sec": 50.0,
    }
    output_root = _value(context, "output_directory")
    runtime_results = _value(context, "results_directory")
    if not runtime_results:
        runtime_results = str(Path(output_root) / f"rviz_{mode}_run")
    publish_visualization = _value(context, "publish_visualization").lower() == "true"
    diagnostics = {
        "diagnostics.enabled": True,
        "diagnostics.publish_visualization": publish_visualization,
        "diagnostics.publish_correspondences_every_n_scans": int(
            _value(context, "publish_correspondences_every_n_scans")
        ),
        "diagnostics.rviz_max_correspondence_lines": int(
            _value(context, "rviz_max_correspondence_lines")
        ),
        "diagnostics.output_directory": output_root,
        "diagnostics.selected_roles": roles,
        "diagnostics.selected_timestamps": timestamps,
        "diagnostics.selected_prediction_poses_xyz_xyzw": poses("pred"),
        "diagnostics.selected_registration_poses_xyz_xyzw": poses("gicp"),
        "diagnostics.selected_inliers": [int(row["num_inliers"]) for row in audit_rows],
        "diagnostics.selected_iterations": [int(row["iterations"]) for row in audit_rows],
        "diagnostics.selected_final_errors": [float(row["final_error"]) for row in audit_rows],
        "diagnostics.selected_registration_runtimes_ms": [float(row["runtime_ms"]) for row in audit_rows],
        "diagnostics.hold_selected_role": selected_role if mode == "selected" else "",
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
            diagnostics,
            {
                "map_path": _value(context, "map_path"),
                "results_directory": runtime_results,
                "max_scans": int(_value(context, "max_scans")),
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
            "ros2", "bag", "play", _value(context, "bag_path"),
            "--clock", "100.0", "--rate", _value(context, "playback_rate"),
            "--delay", "3.0", "--disable-keyboard-controls",
            "--topics", "/odom", "/imu/data", "/velodyne_points",
        ],
        output="screen",
    )
    actions = [adapter, ekf, localizer, base_to_lidar, lidar_to_imu]
    if _value(context, "launch_rviz").lower() == "true":
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="gicp_diagnostic_rviz",
            arguments=["-d", str(share / "rviz" / "gicp_live_diagnostic.rviz")],
            output="screen",
        ))
    if mode == "continuous":
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=localizer,
            on_exit=[EmitEvent(event=Shutdown(reason="0-50 s localizer completed"))],
        )))
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=bag_player,
            on_exit=[TimerAction(
                period=3.0,
                actions=[
                    LogInfo(msg="Bag playback ended; flushing continuous diagnostics."),
                    EmitEvent(event=Shutdown(reason="offline bag completed")),
                ],
            )],
        )))
    actions.append(bag_player)
    return actions


def generate_launch_description():
    root = "/home/a/Desktop/shihoon/bunker_localization_ws"
    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOG_DIR", "/tmp/bunker_localization_ros_logs"),
        DeclareLaunchArgument("mode", default_value="continuous"),
        DeclareLaunchArgument("selected_role", default_value="largest_negative_dz"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("publish_visualization", default_value="true"),
        DeclareLaunchArgument("publish_correspondences_every_n_scans", default_value="1"),
        DeclareLaunchArgument("rviz_max_correspondence_lines", default_value="200"),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("max_scans", default_value="0"),
        DeclareLaunchArgument(
            "bag_path", default_value="/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346"
        ),
        DeclareLaunchArgument(
            "map_path",
            default_value="/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply",
        ),
        DeclareLaunchArgument(
            "output_directory", default_value=f"{root}/results/gicp_correspondence_rviz_diagnostic"
        ),
        DeclareLaunchArgument("results_directory", default_value=""),
        DeclareLaunchArgument(
            "selected_scans_handoff",
            default_value=f"{root}/results/gicp_vertical_observability_diagnostic/rviz_handoff_selected_scans.json",
        ),
        DeclareLaunchArgument(
            "production_localization_csv",
            default_value=f"{root}/results/planar_ekf_gicp_gate/independent_163346_0_50s/localization.csv",
        ),
        OpaqueFunction(function=_setup),
    ])
