import csv
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_surface_pose_z_diagnostic.py"
CONFIG = PACKAGE / "config" / "imu_gicp_surface_pose_z_diagnostic.yaml"
SPEC = importlib.util.spec_from_file_location("surface_pose_z_diagnostic", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


FIT_CONFIG = {
    "huber_parameter": 1.345,
    "maximum_iterations": 100,
    "convergence_tolerance": 1.0e-10,
    "minimum_scale_m": 1.0e-9,
}


def common_rows(a, b, height, *, points=80, seed=3):
    generator = np.random.default_rng(seed)
    rows = []
    for label, offset in ((DIAGNOSTIC.GROUND, 0.0), (DIAGNOSTIC.STAGE, height)):
        x = generator.uniform(-3.0, 3.0, points)
        y = generator.uniform(-2.0, 2.0, points)
        for x_value, y_value in zip(x, y):
            rows.append({
                "x_m": x_value,
                "y_m": y_value,
                "z_m": a * x_value + b * y_value + 0.4 + offset,
                "physical_label": label,
            })
    return rows


def write_candidate_csv(path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "diagnostic_candidate_id", "start_offset_sec", "end_offset_sec", "sample_count",
            "automatic_physical_label", "median_z_m",
        ])
        writer.writeheader()
        for candidate_id in range(1, 23):
            writer.writerow({
                "diagnostic_candidate_id": candidate_id,
                "start_offset_sec": candidate_id,
                "end_offset_sec": candidate_id + 0.5,
                "sample_count": 6,
                "automatic_physical_label": "UNASSIGNED",
                # Deliberately anti-physical ordering: the loader must ignore z for labels.
                "median_z_m": 1000.0 if candidate_id <= 11 else -1000.0,
            })


def fixed_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["fixed_physical_labels"]


def test_exact_common_tilt_synthetic_case_recovers_a_b_and_h():
    fit = DIAGNOSTIC.fit_common_planes(common_rows(0.08, -0.03, 0.15), FIT_CONFIG)
    np.testing.assert_allclose(fit["coefficients"], [0.08, -0.03, 0.4, 0.15], atol=1.0e-10)
    assert fit["rank"] == 4
    assert fit["converged"] is True


def test_nonparallel_synthetic_planes_are_distinguished():
    ground = common_rows(0.05, -0.02, 0.0, points=60)[:60]
    stage = []
    generator = np.random.default_rng(8)
    for x_value, y_value in zip(generator.uniform(-2, 2, 60), generator.uniform(-2, 2, 60)):
        stage.append({
            "x_m": x_value, "y_m": y_value,
            "z_m": -0.12 * x_value + 0.09 * y_value + 0.7,
            "physical_label": DIAGNOSTIC.STAGE,
        })
    ground_json = DIAGNOSTIC.plane_fit_json(
        DIAGNOSTIC.fit_plane(ground, FIT_CONFIG), DIAGNOSTIC.GROUND, "test", 1
    )
    stage_json = DIAGNOSTIC.plane_fit_json(
        DIAGNOSTIC.fit_plane(stage, FIT_CONFIG), DIAGNOSTIC.STAGE, "test", 1
    )
    assert ground_json["a_dz_dx"] == pytest.approx(0.05, abs=1.0e-10)
    assert stage_json["a_dz_dx"] == pytest.approx(-0.12, abs=1.0e-10)
    assert DIAGNOSTIC.parallelism(ground_json, stage_json)["angle_between_normals_deg"] > 5.0


def test_physical_label_lock_is_exact_and_ignores_candidate_z(tmp_path):
    source = tmp_path / "candidates.csv"
    write_candidate_csv(source)
    rows = DIAGNOSTIC.load_fixed_candidates(source, fixed_config())
    assert [row["diagnostic_id"] for row in rows if row["physical_label"] == DIAGNOSTIC.GROUND] == list(range(1, 12))
    assert [row["diagnostic_id"] for row in rows if row["physical_label"] == DIAGNOSTIC.STAGE] == list(range(12, 23))
    assert fixed_config()["ground_after"] == "unavailable"


def test_measured_height_cannot_leak_into_blind_fit():
    assert "measured" not in inspect.signature(DIAGNOSTIC.fit_common_planes).parameters
    assert "measured" not in inspect.signature(DIAGNOSTIC.blind_fit_analysis).parameters
    rows = common_rows(0.02, 0.04, 0.31)
    first = DIAGNOSTIC.fit_common_planes(rows, FIT_CONFIG)["coefficients"]
    unrelated_measured_height = 9.99
    second = DIAGNOSTIC.fit_common_planes(rows, FIT_CONFIG)["coefficients"]
    assert unrelated_measured_height != 0.15
    np.testing.assert_array_equal(first, second)


def test_pair_cherry_pick_cannot_change_labels_or_fit_selection(tmp_path):
    source = tmp_path / "candidates.csv"
    write_candidate_csv(source)
    rows = DIAGNOSTIC.load_fixed_candidates(source, fixed_config())
    assert "pair_diagnostics" not in inspect.signature(DIAGNOSTIC.load_fixed_candidates).parameters
    assert "measured_height" not in inspect.signature(DIAGNOSTIC.load_fixed_candidates).parameters
    assert rows[6]["physical_label"] == DIAGNOSTIC.GROUND
    assert rows[11]["physical_label"] == DIAGNOSTIC.STAGE


def test_major_fit_outputs_are_finite():
    fit = DIAGNOSTIC.fit_common_planes(common_rows(-0.04, 0.06, 0.22), FIT_CONFIG)
    summary = DIAGNOSTIC.common_fit_json(
        fit, "test", {DIAGNOSTIC.GROUND: 11, DIAGNOSTIC.STAGE: 11}
    )
    numeric = [
        summary["a_dz_dx"], summary["b_dz_dy"], summary["c_ground_m"],
        summary["h_stage_minus_ground_m"], summary["slope_angle_deg"],
        summary["rmse_m"], summary["p95_absolute_residual_m"],
        summary["design_matrix_condition_number"],
    ]
    assert np.all(np.isfinite(numeric))


def test_protected_input_fingerprint_and_output_separation(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    source = protected / "source.csv"
    source.write_text("immutable", encoding="utf-8")
    before = DIAGNOSTIC.physical_gate.fingerprint(source, include_hash=True)
    DIAGNOSTIC.physical_gate.validate_output_path(tmp_path / "separate_output", [protected])
    with pytest.raises(ValueError):
        DIAGNOSTIC.physical_gate.validate_output_path(protected / "new_result", [protected])
    assert DIAGNOSTIC.physical_gate.fingerprint(source, include_hash=True) == before
