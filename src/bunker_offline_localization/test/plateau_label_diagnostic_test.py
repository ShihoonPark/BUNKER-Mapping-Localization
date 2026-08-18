import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_plateau_label_diagnostic.py"
CONFIG = PACKAGE / "config" / "imu_gicp_physical_gate.yaml"
SPEC = importlib.util.spec_from_file_location("plateau_label_diagnostic", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def pose_record(index, offset, *, accepted=True, z=0.0, pitch=0.0):
    return {
        "row_index": index,
        "timestamp": offset,
        "offset": offset,
        "accepted": accepted,
        "position": np.asarray([0.1 * offset, 0.0, z]),
        "rpy": np.asarray([0.0, pitch, 0.0]),
    }


def diagnostic_candidate(candidate_id, median_z):
    return {
        "diagnostic_candidate_id": candidate_id,
        "start_offset_sec": float(candidate_id),
        "end_offset_sec": float(candidate_id) + 0.6,
        "median_z_m": median_z,
    }


def test_full_interval_config_is_height_blind_and_assigns_no_labels():
    diagnostic = load_config()["plateau_label_diagnostic"]
    assert diagnostic["valid_start_sec"] == pytest.approx(0.0)
    assert diagnostic["valid_end_sec"] == pytest.approx(50.0)
    assert diagnostic["measured_height_used_for_candidate_detection"] is False
    assert diagnostic["assign_new_physical_labels"] is False
    assert diagnostic["conclusion"] == DIAGNOSTIC.DIAGNOSTIC_CONCLUSION


def test_full_interval_detection_does_not_cross_reject_or_large_gap():
    config = load_config()
    records = []
    # Stable runs occur before 20 s and after 40 s. A rejection and a >0.15 s gap split them.
    for index, offset in enumerate(np.arange(1.0, 2.01, 0.1)):
        records.append(pose_record(index, float(offset)))
    records.append(pose_record(len(records), 2.1, accepted=False))
    start_index = len(records)
    for index, offset in enumerate(np.arange(41.0, 42.01, 0.1), start=start_index):
        records.append(pose_record(index, float(offset), z=0.2))
    candidates, detection = DIAGNOSTIC.detect_full_interval_candidates(records, config)
    assert detection["search_start_sec"] == pytest.approx(0.0)
    assert detection["search_end_sec"] == pytest.approx(50.0)
    assert len(candidates) == 2
    assert candidates[0]["end_offset_sec"] < 20.0
    assert candidates[1]["start_offset_sec"] > 40.0
    assert all(candidate["automatic_physical_label"] == "UNASSIGNED" for candidate in candidates)
    assert all(candidate["used_for_height_selection"] is False for candidate in candidates)


def test_candidate_pair_table_is_post_hoc_and_not_error_ranked():
    candidates = [
        diagnostic_candidate(1, 0.0),
        diagnostic_candidate(2, 0.5),
        diagnostic_candidate(3, 0.16),
    ]
    rows = DIAGNOSTIC.candidate_pair_height_diagnostics(candidates, 0.150)
    assert [(row["candidate_a_id"], row["candidate_b_id"]) for row in rows] == [
        (1, 2), (1, 3), (2, 3)
    ]
    # Pair 1-3 is closest to 0.150 m but remains second in natural ID order.
    assert rows[1]["post_hoc_absolute_error_m"] == pytest.approx(0.01)
    assert all(row["used_for_candidate_selection"] is False for row in rows)
    assert all(row["table_order"] == "ascending_candidate_id_not_measurement_error" for row in rows)


def test_historical_labels_raw_z_order_and_height_fail_are_preserved():
    rows = [
        {"candidate_id": "4", "median_z_m": "0.133460787993"},
        {"candidate_id": "5", "median_z_m": "0.201568099796"},
        {"candidate_id": "6", "median_z_m": "0.268985233451"},
        {"candidate_id": "7", "median_z_m": "0.281900203728"},
    ]
    height = {
        "status": "HEIGHT_FAIL",
        "plateau_selection": {"groups": {
            "stage_top": {"candidate_ids": [4]},
            "ground_after": {"candidate_ids": [5, 6, 7]},
        }},
        "ground_plane": {"slope_angle_deg": 8.20574257033721, "slope_magnitude": 0.144204510829},
    }
    result = DIAGNOSTIC.historical_temporal_labeling(rows, height)
    assert result["preserved_height_status"] == "HEIGHT_FAIL"
    assert result["historical_stage_top_candidate_ids"] == [4]
    assert result["historical_ground_after_candidate_ids"] == [5, 6, 7]
    assert result["raw_median_z_ascending_candidate_ids"] == [4, 5, 6, 7]
    assert result["fitted_ground_plane_slope_deg"] == pytest.approx(8.20574257033721)
    assert "labeling" in result["surface_labeling_warning"]
    assert result["new_physical_labels_assigned"] is False


def test_report_declares_inconclusive_without_changing_preserved_fail(tmp_path):
    summary = {
        "diagnostic_conclusion": DIAGNOSTIC.DIAGNOSTIC_CONCLUSION,
        "candidate_search_window_sec": [0.0, 50.0],
        "candidate_count": 1,
        "measured_height_used_for_candidate_detection": False,
        "new_physical_labels_assigned": False,
        "historical_temporal_labeling": {
            "preserved_height_status": "HEIGHT_FAIL",
            "raw_median_z_ordering_text": "candidate 4 (0.133461 m) < candidate 5 (0.201568 m)",
            "fitted_ground_plane_slope_deg": 8.20574257033721,
            "fitted_ground_plane_slope_m_per_m": 0.144204510829,
            "surface_labeling_warning": "surface labeling suspect",
        },
        "input_integrity": {"unchanged": True},
        "source_height_gate_directory": str(tmp_path / "source"),
        "output_directory": str(tmp_path / "output"),
    }
    candidate = {
        "diagnostic_candidate_id": 1,
        "start_offset_sec": 1.0,
        "end_offset_sec": 2.0,
        "sample_count": 11,
        "median_x_m": 0.0,
        "median_y_m": 0.0,
        "median_z_m": 0.0,
        "median_pitch_deg": 0.0,
    }
    report = DIAGNOSTIC.report_text(summary, [candidate])
    assert "HEIGHT_FAIL" in report
    assert DIAGNOSTIC.DIAGNOSTIC_CONCLUSION in report
    assert "does not interpret it as a production-localization failure" in report
    assert "no closest pair is highlighted" in report
