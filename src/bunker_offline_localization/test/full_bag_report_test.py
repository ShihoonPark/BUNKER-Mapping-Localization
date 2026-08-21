import importlib.util
from pathlib import Path

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_full_bag_report.py"
SPEC = importlib.util.spec_from_file_location("full_bag_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def row(timestamp, accepted=True, x=0.0, yaw_deg=0.0):
    half_yaw = np.deg2rad(yaw_deg) / 2.0
    values = {
        "timestamp": str(timestamp),
        "accepted": "1" if accepted else "0",
        "converged": "1" if accepted else "0",
        "reject_reason": "NONE" if accepted else "NOT_CONVERGED",
        "correction_translation_m": "0.01",
        "correction_roll_rad": "0.0",
        "correction_pitch_rad": "0.0",
        "correction_yaw_rad": "0.0",
        "runtime_ms": "2.0",
    }
    for prefix in ("pred", "gicp"):
        values.update(
            {
                f"{prefix}_x": str(x),
                f"{prefix}_y": "0.0",
                f"{prefix}_z": "0.0",
                f"{prefix}_qx": "0.0",
                f"{prefix}_qy": "0.0",
                f"{prefix}_qz": str(np.sin(half_yaw)),
                f"{prefix}_qw": str(np.cos(half_yaw)),
                f"{prefix}_yaw": str(np.deg2rad(yaw_deg)),
            }
        )
    return values


def test_rejection_runs_preserve_every_maximal_interval():
    rows = [row(0.0), row(0.1, False), row(0.2, False), row(0.3), row(0.4, False)]
    runs = REPORT.find_rejection_runs(rows, np.arange(5) * 0.1)
    assert [run["length_scans"] for run in runs] == [2, 1]
    assert runs[0]["start_offset_sec"] == pytest.approx(0.1)
    assert runs[0]["end_offset_sec"] == pytest.approx(0.2)
    assert runs[1]["reasons"] == {"NOT_CONVERGED": 1}


def test_segment_statistics_use_nonoverlapping_boundaries():
    offsets = np.asarray([0.9, 49.9, 50.0, 100.0, 150.0, 200.0, 219.1])
    rows = [row(value) for value in offsets]
    prediction_offsets, translation, yaw = REPORT.prediction_steps(rows, offsets)
    segments = REPORT.segment_statistics(
        rows,
        offsets,
        prediction_offsets,
        translation,
        yaw,
        np.full(len(rows), 0.01),
        [0.0, 50.0, 100.0, 150.0, 200.0, 219.251],
    )
    assert [segment["processed_scans"] for segment in segments] == [2, 1, 1, 1, 2]
    assert sum(segment["processed_scans"] for segment in segments) == len(rows)


def test_short_replay_comparison_accepts_millimeter_subdegree_nondeterminism():
    short = [row(10.0, x=0.0), row(10.1, x=0.1, yaw_deg=1.0)]
    full = [row(10.0, x=0.004, yaw_deg=0.2), row(10.1, x=0.104, yaw_deg=1.2)]
    comparison = REPORT.compare_short_replay(full, short, origin_timestamp=10.0)
    assert comparison["counts_and_reasons_identical"] is True
    assert comparison["near_identical"] is True
    assert comparison["accepted_pose_translation_difference_m"]["max"] == pytest.approx(0.004)
    assert comparison["accepted_pose_rotation_difference_deg"]["max"] == pytest.approx(0.2)


def test_correction_rotation_uses_full_so3_not_euler_vector_norm():
    values = row(0.0)
    values["correction_yaw_rad"] = str(np.pi / 2.0)
    angle = REPORT.correction_rotation_angles([values])[0]
    assert angle == pytest.approx(np.pi / 2.0)


def test_latency_window_excludes_samples_after_50_seconds():
    rows = [
        {"bag_relative_time": "49.9", "registration_runtime_ms": "2.0", "core_localization_latency_ms": "8.0"},
        {"bag_relative_time": "50.1", "registration_runtime_ms": "50.0", "core_localization_latency_ms": "80.0"},
    ]
    metrics = REPORT.latency_window_metrics(rows)
    assert metrics["sample_count"] == 1
    assert metrics["registration_runtime_ms"]["max"] == 2.0
    assert metrics["core_localization_latency_ms"]["max"] == 8.0
