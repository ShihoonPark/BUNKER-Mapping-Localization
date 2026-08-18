import math
import importlib.util

import generate_gicp_correspondence_rviz_diagnostic as diagnostic
import pytest
import yaml


def load_rviz_launch_module():
    path = (
        diagnostic.ROOT
        / "src/bunker_offline_localization/launch/gicp_rviz_diagnostic.launch.py"
    )
    spec = importlib.util.spec_from_file_location("gicp_rviz_diagnostic_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protected_input_fingerprints_are_stable_across_read_only_audit():
    before = diagnostic.protected_fingerprints()
    after = diagnostic.protected_fingerprints()
    assert before == after
    assert before["small_gicp"]["porcelain_status"] == ""


def test_latency_distribution_is_finite_and_complete():
    summary = diagnostic.distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert math.isfinite(summary["p95"])
    assert summary["max"] == 4.0


def test_rviz_topics_use_explicit_compatible_qos_and_expected_defaults():
    rviz_path = (
        diagnostic.ROOT
        / "src/bunker_offline_localization/rviz/gicp_live_diagnostic.rviz"
    )
    config = yaml.safe_load(rviz_path.read_text(encoding="utf-8"))
    displays = {
        display["Topic"]["Value"]: display
        for display in config["Visualization Manager"]["Displays"]
        if "Topic" in display
    }
    assert displays["/map_cloud"]["Topic"] == {
        "Depth": 1,
        "Durability Policy": "Transient Local",
        "History Policy": "Keep Last",
        "Reliability Policy": "Reliable",
        "Value": "/map_cloud",
    }
    for topic in diagnostic.TOPICS:
        if topic == "/map_cloud":
            continue
        assert displays[topic]["Topic"]["Reliability Policy"] == "Best Effort"
        assert displays[topic]["Topic"]["Durability Policy"] == "Volatile"
    assert displays["/map_cloud"]["Enabled"] is True
    assert displays["/registered_scan"]["Enabled"] is True
    assert displays["/gicp_path"]["Enabled"] is True
    assert displays["/gicp_pose"]["Enabled"] is True
    assert displays["/raw_scan"]["Enabled"] is False
    assert displays["/gicp_correspondences"]["Enabled"] is False


def test_full_bag_mode_disables_window_and_requires_complete_one_x_replay():
    launch = load_rviz_launch_module()
    launch._validate_mode("full_bag", "unused", 1.0, 0)
    full_window = launch._window_parameters("full_bag")
    continuous_window = launch._window_parameters("continuous")
    assert full_window["time_window.enabled"] is False
    assert full_window["time_window.origin_timestamp"] == launch.WINDOW_ORIGIN_TIMESTAMP
    assert continuous_window["time_window.enabled"] is True
    with pytest.raises(RuntimeError, match="playback_rate"):
        launch._validate_mode("full_bag", "unused", 2.0, 0)
    with pytest.raises(RuntimeError, match="max_scans"):
        launch._validate_mode("full_bag", "unused", 1.0, 1)


def test_full_bag_player_waits_for_final_reliable_lidar_delivery_only_in_full_mode():
    launch = load_rviz_launch_module()
    full = launch._bag_player_command("full_bag", "/tmp/read_only_bag", "9.0")
    continuous = launch._bag_player_command(
        "continuous", "/tmp/read_only_bag", "2.0"
    )
    assert full[full.index("--rate") + 1] == "1.0"
    assert full[full.index("--wait-for-all-acked") + 1] == "5000"
    assert continuous[continuous.index("--rate") + 1] == "2.0"
    assert "--wait-for-all-acked" not in continuous
