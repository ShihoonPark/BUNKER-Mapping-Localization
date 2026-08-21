import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("ROS_LOG_DIR", "/tmp/bunker_localization_test_logs")
from launch import LaunchContext


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))
import generate_filter_comparison as comparison


def load_yaml(name):
    with (PACKAGE / "config" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_launch_module():
    path = PACKAGE / "launch" / "filter_comparison.launch.py"
    spec = importlib.util.spec_from_file_location("filter_comparison_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_sensor_fields_and_disabled_imu_orientation():
    parameters = load_yaml("filter_common.yaml")["localization_filter_node"][
        "ros__parameters"
    ]
    legacy_ekf = load_yaml("ekf.yaml")["ekf_filter_node"]["ros__parameters"]
    shared_keys = {
        "frequency", "sensor_timeout", "two_d_mode", "map_frame", "odom_frame",
        "base_link_frame", "world_frame", "odom0", "odom0_config", "odom0_queue_size",
        "odom0_differential", "odom0_relative", "odom0_pose_rejection_threshold",
        "odom0_twist_rejection_threshold", "imu0", "imu0_config", "imu0_queue_size",
        "imu0_differential", "imu0_relative", "imu0_remove_gravitational_acceleration",
        "imu0_angular_velocity_rejection_threshold",
    }
    assert {key: parameters[key] for key in shared_keys} == {
        key: legacy_ekf[key] for key in shared_keys
    }
    assert parameters["odom0_config"] == [
        True, True, False,
        False, False, True,
        True, False, False,
        False, False, True,
        False, False, False,
    ]
    assert parameters["imu0_config"] == [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, True,
        False, False, False,
    ]
    assert not any(parameters["imu0_config"][3:6])
    assert [index for index, enabled in enumerate(parameters["imu0_config"]) if enabled] == [11]


def test_ekf_and_ukf_configs_and_launch_paths_load():
    assert load_yaml("filter_ekf.yaml")["localization_filter_node"]["ros__parameters"] == {}
    ukf = load_yaml("filter_ukf.yaml")["localization_filter_node"]["ros__parameters"]
    assert ukf == {"alpha": 0.001, "kappa": 0.0, "beta": 2.0}

    module = load_launch_module()
    for filter_type in comparison.FILTERS:
        for dataset in comparison.DATASETS:
            context = LaunchContext()
            context.launch_configurations["filter_type"] = filter_type
            context.launch_configurations["dataset"] = dataset
            context.launch_configurations["results_directory"] = "/tmp/filter_test"
            context.launch_configurations["max_scans"] = "1"
            context.launch_configurations["publish_test_identity_base_to_lidar"] = "true"
            actions = module._launch_setup(context, PACKAGE)
            assert any(
                getattr(action, "node_executable", None) == f"{filter_type}_node"
                for action in actions
            )


def test_independent_window_remains_zero_to_fifty_seconds():
    config = load_yaml("independent_163346.yaml")
    for node in ("covariance_adapter", "offline_localizer"):
        window = config[node]["ros__parameters"]["time_window"]
        assert window["enabled"] is True
        assert window["start_offset_sec"] == 0.0
        assert window["end_offset_sec"] == 50.0


def test_prediction_to_registration_delta_metrics():
    prediction_position = np.asarray([[0.0, 0.0, 0.0]])
    prediction_quaternion = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    registration_position = np.asarray([[3.0, 4.0, 0.0]])
    registration_quaternion = np.asarray(
        [[0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]]
    )
    translation, rotation, yaw = comparison.correction_values(
        prediction_position,
        prediction_quaternion,
        registration_position,
        registration_quaternion,
    )
    assert translation[0] == 5.0
    assert math.isclose(rotation[0], 90.0, abs_tol=1.0e-10)
    assert math.isclose(yaw[0], 90.0, abs_tol=1.0e-10)


def test_comparison_distribution_metrics():
    metrics = comparison.distribution([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert metrics["count"] == 4
    assert metrics["min"] == 1.0
    assert metrics["mean"] == 2.5
    assert metrics["median"] == 2.5
    assert math.isclose(metrics["p95"], 3.85)
    assert metrics["max"] == 4.0
