import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_imu_gicp_physical_gate.py"
CONFIG = PACKAGE / "config" / "imu_gicp_physical_gate.yaml"
SPEC = importlib.util.spec_from_file_location("imu_gicp_gate", SCRIPT)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def rotation_record(index, timestamp, rotation, accepted=True):
    return {
        "row_index": index,
        "timestamp": timestamp,
        "offset": timestamp,
        "accepted": accepted,
        "rotation": rotation,
        "position": np.zeros(3),
        "rpy": GATE.rotation_to_rpy(rotation),
    }


def test_so3_log_small_angle():
    vector = np.asarray([1.0e-10, -2.0e-10, 3.0e-10])
    np.testing.assert_allclose(GATE.so3_log(GATE.so3_exp(vector)), vector, atol=1.0e-15)


def test_so3_log_known_ninety_degree_rotation():
    expected = np.asarray([0.0, 0.0, math.pi / 2.0])
    np.testing.assert_allclose(GATE.so3_log(GATE.so3_exp(expected)), expected, atol=1.0e-12)


def test_so3_log_is_stable_near_pi():
    expected = np.asarray([math.pi - 1.0e-7, 0.0, 0.0])
    actual = GATE.so3_log(GATE.so3_exp(expected))
    np.testing.assert_allclose(actual, expected, atol=2.0e-7)


def test_body_frame_relative_rotation_direction():
    map_yaw = GATE.so3_exp([0.0, 0.0, math.pi / 2.0])
    body_roll = GATE.so3_exp([math.pi / 4.0, 0.0, 0.0])
    records = [
        rotation_record(0, 0.0, map_yaw),
        rotation_record(1, 1.0, map_yaw @ body_roll),
    ]
    pairs, excluded = GATE.build_gicp_pairs(records, 1.1)
    assert excluded == {"rejected_endpoint": 0, "nonpositive_dt": 0, "large_gap": 0}
    np.testing.assert_allclose(pairs[0]["omega_gicp"], [math.pi / 4.0, 0.0, 0.0], atol=1.0e-12)


def test_constant_angular_velocity_quaternion_integration():
    rotation = GATE.integrate_constant_rate([0.0, 0.5, 0.0], 2.0)
    np.testing.assert_allclose(GATE.so3_log(rotation), [0.0, 1.0, 0.0], atol=1.0e-10)


def test_gyro_bias_uses_robust_median():
    samples = np.asarray([[0.01, -0.02, 0.03]] * 9 + [[9.0, -8.0, 7.0]])
    result = GATE.robust_gyro_bias(samples)
    np.testing.assert_allclose(result["median_radps"], [0.01, -0.02, 0.03])
    assert not np.allclose(result["mean_radps"], result["median_radps"])


def test_timestamp_interval_time_weighted_average():
    times = np.asarray([0.0, 0.25, 0.75, 1.0])
    values = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    average = GATE.interval_time_weighted_average(times, values, 0.0, 1.0, 0.6, 0.01)
    # Voronoi time weights are [0.125, 0.375, 0.375, 0.125].
    np.testing.assert_allclose(average, [3.0, 0.0, 0.0])


def test_rejected_and_large_gap_pairs_are_never_bridged():
    identity = np.eye(3)
    records = [
        rotation_record(0, 0.0, identity),
        rotation_record(1, 0.1, identity, accepted=False),
        rotation_record(2, 0.2, identity),
        rotation_record(3, 0.5, identity),
        rotation_record(4, 0.6, GATE.so3_exp([0.0, 0.0, 0.1])),
    ]
    pairs, excluded = GATE.build_gicp_pairs(records, 0.15)
    assert [(pair["first_index"], pair["second_index"]) for pair in pairs] == [(3, 4)]
    assert excluded["rejected_endpoint"] == 2
    assert excluded["large_gap"] == 1


def test_axis_correlation_detects_mapping_and_sign():
    time = np.linspace(0.0, 8.0, 500)
    gicp = np.column_stack((np.sin(time), np.cos(0.7 * time), np.sin(1.7 * time)))
    imu = np.column_stack((gicp[:, 0], -gicp[:, 1], gicp[:, 2]))
    arrays = {"synthetic": {"times": time, "gicp": gicp, "imu": imu, "segments": np.zeros(time.size, dtype=int)}}
    rows = GATE.compute_axis_correlation_matrices(arrays)
    expected_y = next(row for row in rows if row["gicp_axis"] == "y" and row["imu_axis"] == "y")
    assert expected_y["best_absolute_mapping"] is True
    assert expected_y["pearson"] < -0.999


def test_lag_search_recovers_synthetic_delayed_signal():
    times = np.arange(0.0, 20.0, 0.01)
    reference = np.sin(1.3 * times) + 0.4 * np.sin(3.1 * times)
    delay = 0.12
    signal = np.sin(1.3 * (times - delay)) + 0.4 * np.sin(3.1 * (times - delay))
    rows, best, zero = GATE.search_lags(
        times, reference, signal, np.zeros(times.size, dtype=int), -0.3, 0.3, 0.01
    )
    assert rows[best]["lag_sec"] == pytest.approx(delay, abs=0.011)
    assert rows[best]["pearson"] > rows[zero]["pearson"]


def synthetic_plane_points(height=0.0, count=80, seed=4):
    generator = np.random.default_rng(seed)
    x = generator.uniform(-2.0, 2.0, count)
    y = generator.uniform(-1.5, 1.5, count)
    z = 0.08 * x - 0.03 * y + 0.7 + height
    return np.column_stack((x, y, z))


def test_robust_ground_plane_fit_rejects_outlier():
    points = synthetic_plane_points()
    points[0, 2] += 8.0
    plane = GATE.robust_plane_fit(points)
    np.testing.assert_allclose(plane["coefficients"], [0.08, -0.03, 0.7], atol=2.0e-3)


def test_known_synthetic_stage_height():
    before = synthetic_plane_points(seed=1)
    after = synthetic_plane_points(seed=2)
    stage = synthetic_plane_points(height=0.42, seed=3)
    result = GATE.estimate_height_from_points(
        before, stage, after,
        {"available": True, "height_m": 0.42, "tolerance_m": 0.01},
        bootstrap_samples=30, seed=7,
    )
    assert result["estimated_stage_height_m"] == pytest.approx(0.42, abs=1.0e-8)
    assert result["status"] == "HEIGHT_PASS"


def test_unavailable_physical_height_gives_pending():
    before = synthetic_plane_points(seed=1)
    after = synthetic_plane_points(seed=2)
    stage = synthetic_plane_points(height=0.42, seed=3)
    result = GATE.estimate_height_from_points(
        before, stage, after,
        {"available": False, "height_m": 0.0, "tolerance_m": 0.05},
    )
    assert result["status"] == "HEIGHT_PENDING"
    assert result["estimated_stage_height_m"] == pytest.approx(0.42, abs=1.0e-8)
    assert result["absolute_error_m"] is None


def test_original_paths_are_configured_read_only_and_output_is_separate(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["inputs"]["bag"] == "/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346"
    assert config["inputs"]["localization_csv"].endswith(
        "results/planar_ekf_gicp_gate/independent_163346_0_50s/localization.csv"
    )
    protected = tmp_path / "production_result"
    protected.mkdir()
    source = protected / "localization.csv"
    source.write_text("unchanged", encoding="utf-8")
    before = GATE.fingerprint(source, include_hash=True)
    with source.open(encoding="utf-8") as stream:
        assert stream.read() == "unchanged"
    assert GATE.fingerprint(source, include_hash=True) == before
    with pytest.raises(ValueError):
        GATE.validate_output_path(protected / "new_output", [protected])
    GATE.validate_output_path(tmp_path / "separate_output", [protected])
