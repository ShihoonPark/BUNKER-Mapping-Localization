import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/bunker_offline_localization"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bag_d_config_uses_metadata_window_without_legacy_staging_seed():
    config = yaml.safe_load(
        (PACKAGE / "config/independent_bag_D_20260819.yaml").read_text(
            encoding="utf-8"
        )
    )
    adapter = config["covariance_adapter"]["ros__parameters"]["time_window"]
    localizer = config["offline_localizer"]["ros__parameters"]
    assert adapter["origin_timestamp"] == pytest.approx(1787142248.3215761)
    assert adapter["start_offset_sec"] == 0.0
    assert adapter["end_offset_sec"] == 50.0
    assert "translation" not in localizer["initialization"]
    assert "rotation_xyzw" not in localizer["initialization"]


def test_production_ekf_zeroes_arbitrary_odom_origin_without_relaxing_gates():
    ekf = yaml.safe_load((PACKAGE / "config/ekf.yaml").read_text(encoding="utf-8"))[
        "ekf_filter_node"
    ]["ros__parameters"]
    comparison = yaml.safe_load(
        (PACKAGE / "config/filter_common.yaml").read_text(encoding="utf-8")
    )["localization_filter_node"]["ros__parameters"]
    for parameters in (ekf, comparison):
        assert parameters["odom0_relative"] is True
        assert parameters["odom0_differential"] is False
        assert parameters["odom0_pose_rejection_threshold"] == 5.0
        assert parameters["odom0_twist_rejection_threshold"] == 5.0


def test_bag_d_launch_loads_ranked_seed_and_preserves_short_full_policy(tmp_path):
    launch = load_module(
        PACKAGE / "launch/bag_d_localization.launch.py", "bag_d_launch"
    )
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "success": True,
                "best_candidate": {
                    "T_map_lidar": {
                        "translation": [1.0, 2.0, 3.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parameters = launch._seed_parameters(seed_path)
    assert parameters["initialization.translation"] == [1.0, 2.0, 3.0]
    assert parameters["initialization.rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert launch._window_parameters("short")["time_window.enabled"] is True
    assert launch._window_parameters("full")["time_window.enabled"] is False
    short_command = launch._bag_command("short", "/tmp/read_only_bag")
    full_command = launch._bag_command("full", "/tmp/read_only_bag")
    assert "--wait-for-all-acked" not in short_command
    assert full_command[full_command.index("--wait-for-all-acked") + 1] == "5000"
    with pytest.raises(RuntimeError, match="short or full"):
        launch._validate_mode("continuous")


def test_generic_independent_report_rpy_conversion_uses_ros_xyzw_convention():
    report = load_module(
        PACKAGE / "scripts/generate_independent_report.py", "independent_report"
    )
    half = np.sqrt(0.5)
    rpy = report.quaternion_roll_pitch_yaw(
        np.asarray([[0.0, 0.0, half, half]])
    )
    assert rpy.shape == (1, 3)
    assert rpy[0, 0] == pytest.approx(0.0)
    assert rpy[0, 1] == pytest.approx(0.0)
    assert rpy[0, 2] == pytest.approx(np.pi / 2.0)


def test_bag_d_rviz_launch_reuses_production_seed_and_full_bag_accounting(tmp_path):
    production = load_module(
        PACKAGE / "launch/bag_d_localization.launch.py", "bag_d_production_launch"
    )
    visualization = load_module(
        PACKAGE / "launch/bag_d_rviz_localization.launch.py", "bag_d_rviz_launch"
    )
    seed_path = tmp_path / "coarse_seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "success": True,
                "best_candidate": {
                    "T_map_lidar": {
                        "translation": [3.0, 4.0, 0.2],
                        "rotation_xyzw": [0.0, 0.0, 1.0, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert visualization._seed_parameters(seed_path) == production._seed_parameters(
        seed_path
    )
    assert visualization.INDEPENDENT_CONFIG == "independent_bag_D_20260819.yaml"
    assert visualization.EKF_CONFIG == "ekf.yaml"
    assert visualization._window_parameters()["time_window.enabled"] is False
    full = visualization._full_bag_parameters(1.0, 0)
    assert full["full_bag.enabled"] is True
    assert full["full_bag.expected_lidar_inputs"] == 2181
    assert full["full_bag.replay_rate"] == 1.0


def test_bag_d_rviz_fast_replay_is_visualization_only_and_preserves_extrinsics():
    launch = load_module(
        PACKAGE / "launch/bag_d_rviz_localization.launch.py", "bag_d_rviz_fast_launch"
    )
    assert launch._production_verification_mode(1.0, 0) is True
    assert launch._production_verification_mode(2.0, 0) is False
    assert launch._production_verification_mode(3.0, 0) is False
    assert launch._production_verification_mode(1.0, 40) is False
    command = launch._bag_command("/tmp/read_only_bag", 3.0, True)
    assert command[command.index("--rate") + 1] == "3.0"
    assert command[command.index("--wait-for-all-acked") + 1] == "5000"
    assert launch.BASE_TO_LIDAR_ARGUMENTS[launch.BASE_TO_LIDAR_ARGUMENTS.index("--z") + 1] == "0"
    assert launch.LIDAR_TO_IMU_ARGUMENTS[launch.LIDAR_TO_IMU_ARGUMENTS.index("--z") + 1] == "-0.07"
