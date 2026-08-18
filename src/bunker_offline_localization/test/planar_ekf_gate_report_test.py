import importlib.util
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_planar_ekf_gate_report.py"
SPEC = importlib.util.spec_from_file_location("planar_gate_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def row(timestamp, z, roll, pitch, yaw, accepted=True):
    values = {
        "timestamp": str(timestamp),
        "accepted": "1" if accepted else "0",
        "converged": "1" if accepted else "0",
        "reject_reason": "NONE" if accepted else "NOT_CONVERGED",
        "runtime_ms": "2.0",
        "correction_translation_m": "0.1",
        "correction_roll_rad": "0.01",
        "correction_pitch_rad": "-0.02",
        "correction_yaw_rad": "0.03",
    }
    for prefix in ("pred", "gicp"):
        values.update({
            f"{prefix}_x": "1.0", f"{prefix}_y": "2.0", f"{prefix}_z": str(z),
            f"{prefix}_qx": "0.0", f"{prefix}_qy": "0.0", f"{prefix}_qz": "0.0",
            f"{prefix}_qw": "1.0", f"{prefix}_roll": str(roll),
            f"{prefix}_pitch": str(pitch), f"{prefix}_yaw": str(yaw),
        })
    return values


def test_dataset_metrics_verify_explicit_corrections_and_six_dof():
    rows = [row(25.0, 0.0, 0.1, -0.1, 0.0), row(26.0, 0.0, 0.1, -0.1, 0.2)]
    metrics, _ = REPORT.evaluate_dataset(rows, origin_timestamp=0.0, ramp_window=(25.0, 26.0))
    assert metrics["accepted_six_dof_finite"] == 2
    assert metrics["all_attempted_corrections_finite"] is True
    assert metrics["prediction_quaternion_norm_error_max"] == 0.0
    assert metrics["accepted_correction"]["translation_m"]["mean"] == 0.1


def test_dataset_metrics_detect_zero_unobserved_state_hold_error():
    first = row(1.0, 0.3, 0.2, -0.1, 0.0)
    second = row(2.0, 0.3, 0.2, -0.1, 0.4)
    metrics, _ = REPORT.evaluate_dataset([first, second])
    assert metrics["prediction_hold_error"]["z_m"]["max"] == 0.0
    assert metrics["prediction_hold_error"]["roll_rad"]["max"] == 0.0
    assert metrics["prediction_hold_error"]["pitch_rad"]["max"] == 0.0
