import math

import generate_gicp_correspondence_rviz_diagnostic as diagnostic
import yaml


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
