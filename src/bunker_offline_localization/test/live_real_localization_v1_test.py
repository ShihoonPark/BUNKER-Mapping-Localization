import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/bunker_offline_localization"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_config_disables_offline_time_and_enables_accepted_pose_only():
    config = yaml.safe_load(
        (PACKAGE / "config/live_real_localization_v1.yaml").read_text(encoding="utf-8")
    )
    adapter = config["covariance_adapter"]["ros__parameters"]
    ekf = config["ekf_filter_node"]["ros__parameters"]
    localizer = config["offline_localizer"]["ros__parameters"]
    coarse = config["coarse_global_initializer"]["ros__parameters"]
    assert all(parameters["use_sim_time"] is False for parameters in (adapter, ekf, localizer, coarse))
    assert adapter["time_window"]["enabled"] is False
    assert localizer["time_window"]["enabled"] is False
    assert localizer["full_bag"]["enabled"] is False
    assert localizer["max_scans"] == 0
    assert localizer["initialization"]["mode"] == "parameter"
    assert localizer["diagnostics"]["publish_accepted_pose"] is True
    assert localizer["diagnostics"]["publish_visualization"] is False
    assert localizer["diagnostics"]["publish_correspondences_every_n_scans"] == 0
    assert localizer["base_to_lidar"]["available"] is False
    assert localizer["base_to_lidar"]["allow_identity_for_phase1_smoke_test"] is True


def test_live_seed_validation_preserves_t_map_lidar(tmp_path):
    launch = load_module(
        PACKAGE / "launch/live_real_localization_v1.launch.py", "live_v1_launch"
    )
    map_path = tmp_path / "map.ply"
    map_path.touch()
    seed_path = tmp_path / "coarse_initialization.json"
    seed_path.write_text(
        json.dumps(
            {
                "success": True,
                "map_path": str(map_path),
                "scans_used": 1,
                "transform_convention": "p_map = T_map_lidar * p_lidar",
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
    parameters = launch._seed_parameters(seed_path, map_path)
    assert parameters["initialization.translation"] == [1.0, 2.0, 3.0]
    assert parameters["initialization.rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(RuntimeError, match="map mismatch"):
        launch._seed_parameters(seed_path, tmp_path / "other.ply")


def test_live_launches_never_reference_bag_clock_or_cmd_vel():
    for name in ("live_coarse_initializer_v1.launch.py", "live_real_localization_v1.launch.py"):
        source = (PACKAGE / "launch" / name).read_text(encoding="utf-8")
        assert "ros2 bag" not in source
        assert "--clock" not in source
        assert '"/cmd_vel"' not in source
        assert "ExecuteProcess" not in source


def test_canonical_pose_is_map_frame_accepted_only_output():
    source = (PACKAGE / "src/gicp_diagnostics.cpp").read_text(encoding="utf-8")
    assert '"/localization/pose"' in source
    assert "record.accepted && accepted_pose_publisher_" in source
    assert "publishPose(record.registration, raw_message.header.stamp" in source
    assert '"/gicp_status"' in source
    header = (PACKAGE / "include/bunker_offline_localization/gicp_diagnostics.hpp").read_text(
        encoding="utf-8"
    )
    assert "publish_accepted_pose{false}" in header


def test_stationary_csv_analysis_reports_runtime_and_pose_stability(tmp_path, capsys):
    tool = load_module(
        PACKAGE / "scripts/live_real_localization_v1_preflight.py", "live_v1_tool"
    )
    path = tmp_path / "localization.csv"
    path.write_text(
        "accepted,prediction_available,finite_points,runtime_ms,correction_translation_m,"
        "correction_yaw_rad,gicp_x,gicp_y,gicp_yaw\n"
        "1,1,100,10,0.1,0.01,1.0,2.0,3.13\n"
        "0,1,100,20,0.2,-0.02,9.0,9.0,0.0\n"
        "1,1,100,30,0.3,0.03,1.2,2.2,-3.13\n",
        encoding="utf-8",
    )
    assert tool.analyze_csv(path) == 0
    output = capsys.readouterr().out
    assert "accepted_pose_rate=0.666667" in output
    assert "gicp_runtime_ms mean=20.000" in output
    assert "accepted_gicp_x range=0.200000" in output
