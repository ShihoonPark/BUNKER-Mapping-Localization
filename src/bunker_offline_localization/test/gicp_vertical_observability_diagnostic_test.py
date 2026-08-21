import csv
import math
from pathlib import Path

import numpy as np

import generate_gicp_vertical_observability_diagnostic as diagnostic


WORKSPACE = Path("/home/a/Desktop/shihoon/bunker_localization_ws")
CONFIG_PATH = (
    WORKSPACE
    / "src/bunker_offline_localization/config/gicp_vertical_observability_diagnostic.yaml"
)


def analysis_config():
    return {
        "source_parameter_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
        "symmetrize_for_analysis": True,
        "eigenvalue_relative_tolerance": 1.0e-10,
        "eigenvalue_absolute_tolerance": 1.0e-9,
        "pseudo_inverse_rcond": 1.0e-10,
        "coupling_denominator_tolerance": 1.0e-12,
        "symmetry_relative_warning_threshold": 1.0e-10,
        "anchor_z_invariant_tolerance_m": 1.0e-9,
        "vertical_fraction_translation_norm_tolerance_m": 1.0e-12,
    }


def analyze(matrix):
    return diagnostic.analyze_hessian(matrix, np.eye(3), analysis_config())


def anchor_record(index, accepted, prediction_z, gicp_z, group="baseline"):
    return {
        "row_index": index,
        "timestamp": float(index),
        "accepted": accepted,
        "prediction_z": prediction_z,
        "gicp_z": gicp_z,
        "per_scan_delta_z": gicp_z - prediction_z,
        "correction_translation_norm": abs(gicp_z - prediction_z),
        "diagnostic_group": group,
        "hessian": np.eye(6),
        "gicp_rotation": np.eye(3),
    }


def association_records():
    return [
        {
            "row_index": index,
            "timestamp": float(index),
            "bag_relative_time": float(index),
            "diagnostic_candidate_id": index + 1,
            "diagnostic_group": "baseline",
            "accepted": True,
            "gicp_x": x,
            "gicp_y": y,
            "gicp_z": 0.0,
        }
        for index, (x, y) in enumerate(((0.05, 0.0), (1.05, 0.0), (2.05, 0.0)))
    ]


def test_hessian_ordering_fixture_resolves_tz_index():
    order = ["rx", "ry", "rz", "tx", "ty", "tz"]
    matrix = np.diag([11.0, 12.0, 13.0, 14.0, 15.0, 0.25])
    z_index = diagnostic.resolve_parameter_index(order, "tz")
    assert z_index == 5
    assert matrix[z_index, z_index] == 0.25


def test_small_asymmetry_is_symmetrized_and_finite():
    matrix = np.diag([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    matrix[0, 5] = 1.0e-7
    result = analyze(matrix)
    expected = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert result["hessian_valid"]
    assert np.all(np.isfinite([result[f"lambda_{index}"] for index in range(1, 7)]))
    assert np.allclose([result[f"lambda_{index}"] for index in range(1, 7)], expected)


def test_known_weak_z_hessian_has_low_effective_information():
    result = analyze(np.diag([20.0, 20.0, 20.0, 20.0, 20.0, 0.02]))
    assert math.isclose(result["map_vertical_hzz_raw"], 0.02)
    assert math.isclose(result["map_vertical_effective_information"], 0.02)
    assert result["map_vertical_effective_information"] < result["lambda_2"]


def test_z_pitch_coupling_reduces_schur_information():
    matrix = np.diag([10.0] * 6)
    matrix[1, 5] = matrix[5, 1] = 9.0
    result = analyze(matrix)
    assert math.isclose(result["z_pitch_hessian_coupling"], 0.9)
    assert math.isclose(result["map_vertical_effective_information"], 1.9)
    assert result["map_vertical_effective_information"] < result["map_vertical_hzz_raw"]


def test_weakest_eigenvector_z_participation_is_detected():
    result = analyze(np.diag([10.0, 11.0, 12.0, 13.0, 14.0, 0.01]))
    assert math.isclose(result["weakest_mode_z_participation"], 1.0)
    assert result["weakest_mode_roll_participation"] == 0.0
    assert result["weakest_mode_pitch_participation"] == 0.0


def test_anchor_invariant_skips_rejected_pose_as_anchor():
    records = [
        anchor_record(0, True, 0.0, -0.01),
        anchor_record(1, False, -0.01, 4.0),
        anchor_record(2, True, -0.01, -0.03),
    ]
    audit = diagnostic.apply_anchor_and_hessian_metrics(records, analysis_config())
    assert audit["violation_count"] == 0
    assert records[1]["previous_accepted_anchor_z"] == -0.01
    assert records[2]["previous_accepted_anchor_z"] == -0.01


def test_cumulative_accepted_correction_is_exact():
    records = [
        anchor_record(0, True, 0.0, -0.01),
        anchor_record(1, True, -0.01, -0.03),
        anchor_record(2, True, -0.03, -0.06),
    ]
    diagnostic.apply_anchor_and_hessian_metrics(records, analysis_config())
    assert np.allclose(
        [record["cumulative_accepted_delta_z"] for record in records],
        [-0.01, -0.03, -0.06],
    )


def test_mapping_association_is_independent_of_mapping_z():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    trajectory_a = np.column_stack((np.arange(3), xy, np.zeros(3), np.zeros((3, 3)), np.ones(3)))
    trajectory_b = trajectory_a.copy()
    trajectory_b[:, 3] = [1000.0, -2000.0, 3000.0]
    rows_a = diagnostic.associate_mapping(association_records(), trajectory_a, 0.6)
    rows_b = diagnostic.associate_mapping(association_records(), trajectory_b, 0.6)
    assert [row["mapping_index"] for row in rows_a] == [0, 1, 2]
    assert [row["mapping_index"] for row in rows_a] == [
        row["mapping_index"] for row in rows_b
    ]


def test_real_data_metrics_are_finite_and_inputs_unchanged(tmp_path):
    summary = diagnostic.run(CONFIG_PATH, tmp_path / "vertical_observability")
    assert summary["protected_inputs_unchanged"]
    assert summary["input_integrity"]["before"] == summary["input_integrity"]["after"]
    assert summary["input_counts"]["accepted_rows"] == 486
    assert summary["vertical_observability"]["hessian_sanity"]["finite_count"] == 486
    metrics_path = tmp_path / "vertical_observability/gicp_vertical_scan_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    major_fields = (
        "prediction_z", "gicp_z", "correction_z", "cumulative_accepted_delta_z",
        "mapping_z", "independent_minus_mapping_z",
    )
    assert len(rows) == 486
    assert all(math.isfinite(float(row[field])) for row in rows for field in major_fields)
