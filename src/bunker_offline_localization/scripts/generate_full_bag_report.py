#!/usr/bin/env python3

"""Generate a map-relative, full-bag localization stability report.

This evaluator deliberately does not calculate absolute localization error.  It
checks input accounting, temporal continuity, registration quality, and a
short-window replay regression for an independent drive without ground truth.
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_localization_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_report import read_binary_ply_xyz, read_localization_csv, read_tum


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def row_position(row, prefix):
    return np.asarray([_float(row, f"{prefix}_{axis}") for axis in "xyz"])


def row_quaternion(row, prefix):
    quaternion = np.asarray(
        [_float(row, f"{prefix}_q{axis}") for axis in "xyzw"], dtype=float
    )
    norm = np.linalg.norm(quaternion)
    if not np.all(np.isfinite(quaternion)) or norm <= 1.0e-12:
        return np.full(4, np.nan)
    return quaternion / norm


def quaternion_step_angles(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.shape[0] < 2:
        return np.empty(0)
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def quaternion_roll_pitch_yaw(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    x, y, z, w = quaternions.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.column_stack((roll, pitch, yaw))


def correction_rotation_angles(rows):
    """Return shortest SO(3) angles for logged roll/pitch/yaw corrections."""
    roll = np.asarray([_float(row, "correction_roll_rad") for row in rows])
    pitch = np.asarray([_float(row, "correction_pitch_rad") for row in rows])
    yaw = np.asarray([_float(row, "correction_yaw_rad") for row in rows])
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    norms = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    valid = np.isfinite(norms) & (norms > 1.0e-12)
    angles = np.full(norms.shape, np.nan)
    angles[valid] = 2.0 * np.arccos(np.clip(np.abs(qw[valid] / norms[valid]), 0.0, 1.0))
    return angles


def find_rejection_runs(rows, offsets):
    """Find every maximal consecutive run of rejected processed scans."""
    runs = []
    start = None
    for index, row in enumerate(rows):
        rejected = row.get("accepted") != "1"
        if rejected and start is None:
            start = index
        end_of_run = start is not None and (
            not rejected or index == len(rows) - 1
        )
        if not end_of_run:
            continue
        end = index if rejected else index - 1
        run_rows = rows[start : end + 1]
        runs.append(
            {
                "run_id": len(runs) + 1,
                "start_processed_index": start + 1,
                "end_processed_index": end + 1,
                "length_scans": end - start + 1,
                "start_offset_sec": float(offsets[start]),
                "end_offset_sec": float(offsets[end]),
                "duration_sec": float(offsets[end] - offsets[start]),
                "reasons": dict(sorted(Counter(r["reject_reason"] for r in run_rows).items())),
            }
        )
        start = None
    return runs


def prediction_steps(rows, offsets):
    positions = np.asarray([row_position(row, "pred") for row in rows])
    yaws = np.asarray([_float(row, "pred_yaw") for row in rows])
    if len(rows) < 2:
        return np.empty(0), np.empty(0), np.empty(0)
    translation = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    yaw_difference = np.diff(yaws)
    yaw = np.abs(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference)))
    return np.asarray(offsets[1:]), translation, yaw


def segment_statistics(rows, offsets, prediction_offsets, prediction_translation,
                       prediction_yaw, correction_translation, segment_edges):
    """Compute half-open segment statistics, with the final interval closed."""
    segments = []
    for index in range(len(segment_edges) - 1):
        start, end = segment_edges[index], segment_edges[index + 1]
        last = index == len(segment_edges) - 2
        row_mask = (offsets >= start) & (
            (offsets <= end + 1.0e-9) if last else (offsets < end)
        )
        step_mask = (prediction_offsets >= start) & (
            (prediction_offsets <= end + 1.0e-9) if last else (prediction_offsets < end)
        )
        selected = [row for row, keep in zip(rows, row_mask) if keep]
        rejected = [row for row in selected if row.get("accepted") != "1"]
        accepted = len(selected) - len(rejected)
        translation_values = prediction_translation[step_mask]
        yaw_values = prediction_yaw[step_mask]
        correction_values = np.asarray(correction_translation)[row_mask]
        segments.append(
            {
                "label": f"{start:g}-{end:g} s" if not last else f"{start:g}-EOF",
                "start_offset_sec": float(start),
                "end_offset_sec": float(end),
                "processed_scans": len(selected),
                "accepted_scans": accepted,
                "rejected_scans": len(rejected),
                "acceptance_rate": accepted / len(selected) if selected else None,
                "reject_reason_counts": dict(
                    sorted(Counter(row["reject_reason"] for row in rejected).items())
                ),
                "prediction_translation_max_m": (
                    float(np.nanmax(translation_values)) if translation_values.size else None
                ),
                "prediction_yaw_max_deg": (
                    float(np.degrees(np.nanmax(yaw_values))) if yaw_values.size else None
                ),
                "gicp_correction_translation_max_m": (
                    float(np.nanmax(correction_values)) if correction_values.size else None
                ),
            }
        )
    return segments


def _subset_metrics(rows, origin_timestamp, end_offset=50.0):
    selected = [
        row for row in rows
        if _float(row, "timestamp") - origin_timestamp <= end_offset + 1.0e-6
    ]
    offsets = np.asarray([_float(row, "timestamp") - origin_timestamp for row in selected])
    _, prediction_translation, prediction_yaw = prediction_steps(selected, offsets)
    correction = np.asarray([_float(row, "correction_translation_m") for row in selected])
    runtime = np.asarray([_float(row, "runtime_ms") for row in selected])
    rejected = [row for row in selected if row.get("accepted") != "1"]
    return {
        "processed_scans": len(selected),
        "accepted_scans": sum(row.get("accepted") == "1" for row in selected),
        "rejected_scans": len(rejected),
        "reject_reason_counts": dict(
            sorted(Counter(row["reject_reason"] for row in rejected).items())
        ),
        "prediction_translation_m": distribution(prediction_translation),
        "prediction_yaw_deg": distribution(np.degrees(prediction_yaw)),
        "correction_translation_m": distribution(correction),
        "registration_runtime_ms": distribution(runtime),
    }


def compare_short_replay(full_rows, short_rows, origin_timestamp):
    full_metrics = _subset_metrics(full_rows, origin_timestamp)
    short_metrics = _subset_metrics(short_rows, origin_timestamp)
    full_first = [
        row for row in full_rows
        if _float(row, "timestamp") - origin_timestamp <= 50.0 + 1.0e-6
    ]
    short_by_timestamp = {row["timestamp"]: row for row in short_rows}
    position_differences = []
    rotation_differences = []
    status_mismatches = 0
    matched = 0
    for full_row in full_first:
        short_row = short_by_timestamp.get(full_row["timestamp"])
        if short_row is None:
            continue
        matched += 1
        if (
            full_row.get("accepted") != short_row.get("accepted")
            or full_row.get("reject_reason") != short_row.get("reject_reason")
        ):
            status_mismatches += 1
        if full_row.get("accepted") == "1" and short_row.get("accepted") == "1":
            position_differences.append(
                np.linalg.norm(row_position(full_row, "gicp") - row_position(short_row, "gicp"))
            )
            q_full = row_quaternion(full_row, "gicp")
            q_short = row_quaternion(short_row, "gicp")
            rotation_differences.append(
                2.0 * np.arccos(np.clip(abs(np.dot(q_full, q_short)), 0.0, 1.0))
            )
    counts_match = all(
        full_metrics[key] == short_metrics[key]
        for key in ("processed_scans", "accepted_scans", "rejected_scans", "reject_reason_counts")
    )
    max_position = max(position_differences, default=float("nan"))
    max_rotation_deg = np.degrees(max(rotation_differences, default=float("nan")))
    near_identical = bool(
        counts_match
        and matched == len(full_first) == len(short_rows)
        and status_mismatches == 0
        and max_position <= 0.005
        and max_rotation_deg <= 0.25
    )
    return {
        "comparison_window_sec": [0.0, 50.0],
        "full_first_50s": full_metrics,
        "post_fix_short_baseline": short_metrics,
        "matched_timestamps": matched,
        "status_mismatches": status_mismatches,
        "accepted_pose_translation_difference_m": distribution(position_differences),
        "accepted_pose_rotation_difference_deg": distribution(
            np.degrees(rotation_differences)
        ),
        "near_identical_thresholds": {
            "maximum_translation_difference_m": 0.005,
            "maximum_rotation_difference_deg": 0.25,
        },
        "counts_and_reasons_identical": counts_match,
        "near_identical": near_identical,
    }


def latency_window_metrics(rows, end_offset_sec=50.0):
    selected = [
        row for row in rows
        if _float(row, "bag_relative_time") <= end_offset_sec + 1.0e-6
    ]
    return {
        "sample_count": len(selected),
        "registration_runtime_ms": distribution(
            [_float(row, "registration_runtime_ms") for row in selected]
        ),
        "core_localization_latency_ms": distribution(
            [_float(row, "core_localization_latency_ms") for row in selected]
        ),
    }


def _load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_line(metric, unit=""):
    suffix = f" {unit}" if unit else ""
    return (
        f"{metric['mean']:.6f} / {metric['p95']:.6f} / "
        f"{metric['p99']:.6f} / {metric['max']:.6f}{suffix}"
    )


def evaluate(args):
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_localization_csv(args.localization_csv)
    latency_rows = _load_csv(args.latency_csv)
    trajectory_times, trajectory_positions, trajectory_quaternions = read_tum(
        args.estimated_trajectory
    )
    full_accounting = json.loads(Path(args.full_bag_summary).read_text(encoding="utf-8"))
    if not rows or trajectory_times.size == 0:
        raise ValueError("full-bag inputs contain no processed or accepted poses")

    timestamps = np.asarray([_float(row, "timestamp") for row in rows])
    offsets = timestamps - args.origin_timestamp
    accepted_flags = np.asarray([row.get("accepted") == "1" for row in rows])
    converged_flags = np.asarray([row.get("converged") == "1" for row in rows])
    accepted_rows = [row for row in rows if row.get("accepted") == "1"]
    rejected_rows = [row for row in rows if row.get("accepted") != "1"]
    if len(rows) != args.expected_scans:
        raise ValueError(f"expected {args.expected_scans} processed scans, got {len(rows)}")
    if len(accepted_rows) != trajectory_times.size:
        raise ValueError("accepted CSV count does not match TUM pose count")
    if len(latency_rows) != len(rows):
        raise ValueError("latency and localization CSV row counts differ")

    pred_offsets, pred_translation, pred_yaw = prediction_steps(rows, offsets)
    prediction = {
        "translation_step_m": distribution(pred_translation),
        "yaw_step_deg": distribution(np.degrees(pred_yaw)),
        "over_0_5m_count": int(np.count_nonzero(pred_translation > 0.5)),
        "over_1_0m_count": int(np.count_nonzero(pred_translation > 1.0)),
        "over_20deg_count": int(np.count_nonzero(np.degrees(pred_yaw) > 20.0)),
        "over_30deg_count": int(np.count_nonzero(np.degrees(pred_yaw) > 30.0)),
        "largest_translation_offset_sec": float(pred_offsets[int(np.nanargmax(pred_translation))]),
        "largest_yaw_offset_sec": float(pred_offsets[int(np.nanargmax(pred_yaw))]),
    }
    startup_mask = pred_offsets <= 15.0
    old_failure_mask = (pred_offsets >= 8.0) & (pred_offsets <= 10.5)
    prediction["startup_discontinuity_screen"] = {
        "window_sec": [0.0, 15.0],
        "translation_over_1m_count": int(np.count_nonzero(pred_translation[startup_mask] > 1.0)),
        "yaw_over_30deg_count": int(
            np.count_nonzero(np.degrees(pred_yaw[startup_mask]) > 30.0)
        ),
        "previous_failure_window_sec": [8.0, 10.5],
        "previous_failure_window_translation_max_m": float(
            np.nanmax(pred_translation[old_failure_mask])
        ),
        "previous_failure_window_yaw_max_deg": float(
            np.degrees(np.nanmax(pred_yaw[old_failure_mask]))
        ),
        "recurrence_detected": bool(
            np.any(pred_translation[startup_mask] > 1.0)
            or np.any(np.degrees(pred_yaw[startup_mask]) > 30.0)
        ),
    }

    correction_translation = np.asarray(
        [_float(row, "correction_translation_m") for row in rows]
    )
    correction_rotation = correction_rotation_angles(rows)
    inliers = np.asarray([_float(row, "num_inliers") for row in rows])
    final_error = np.asarray([_float(row, "final_error") for row in rows])
    error_per_inlier = np.divide(
        final_error, inliers, out=np.full(final_error.shape, np.nan), where=inliers > 0
    )
    iterations = np.asarray([_float(row, "iterations") for row in rows])
    runtime = np.asarray([_float(row, "runtime_ms") for row in rows])
    core_latency = np.asarray(
        [_float(row, "core_localization_latency_ms") for row in latency_rows]
    )

    accepted_dt = np.diff(trajectory_times)
    accepted_translation = np.linalg.norm(np.diff(trajectory_positions, axis=0), axis=1)
    accepted_rotation = quaternion_step_angles(trajectory_quaternions)
    accepted_step_offsets = trajectory_times[1:] - args.origin_timestamp
    contiguous = accepted_dt <= 0.25
    catastrophic_translation = (accepted_translation > 1.0) & contiguous
    catastrophic_rotation = (np.degrees(accepted_rotation) > 30.0) & contiguous
    trajectory_rpy = quaternion_roll_pitch_yaw(trajectory_quaternions)
    finite_accepted = np.all(np.isfinite(trajectory_positions), axis=1) & np.all(
        np.isfinite(trajectory_quaternions), axis=1
    )

    map_points = read_binary_ply_xyz(args.map_ply)
    map_min = np.min(map_points[:, :2], axis=0)
    map_max = np.max(map_points[:, :2], axis=0)
    inside_bbox = np.all(
        (trajectory_positions[:, :2] >= map_min) & (trajectory_positions[:, :2] <= map_max),
        axis=1,
    )
    rejection_runs = find_rejection_runs(rows, offsets)
    longest_run = max((run["length_scans"] for run in rejection_runs), default=0)
    segments = segment_statistics(
        rows,
        offsets,
        pred_offsets,
        pred_translation,
        pred_yaw,
        correction_translation,
        [0.0, 50.0, 100.0, 150.0, 200.0, args.metadata_duration_sec],
    )
    short_comparison = compare_short_replay(
        rows, read_localization_csv(args.short_localization_csv), args.origin_timestamp
    )
    short_comparison["timing"] = {
        "full_first_50s": latency_window_metrics(latency_rows),
        "post_fix_short_baseline": latency_window_metrics(
            _load_csv(args.short_latency_csv)
        ),
    }

    final_accepted_to_last_lidar = (
        full_accounting["last_lidar_bag_relative_sec"]
        - full_accounting["final_accepted_bag_relative_sec"]
    )
    minimum_segment_acceptance = min(
        segment["acceptance_rate"] for segment in segments if segment["processed_scans"]
    )
    gate_checks = {
        "eof_and_all_inputs_accounted": bool(
            full_accounting.get("bag_eof_received")
            and full_accounting.get("process_survived_to_bag_eof")
            and full_accounting.get("all_lidar_inputs_accounted")
        ),
        "no_large_prediction_discontinuity": bool(
            prediction["over_1_0m_count"] == 0 and prediction["over_30deg_count"] == 0
        ),
        "no_rejection_cascade": bool(longest_run <= 5),
        "no_catastrophic_accepted_jump": bool(
            np.count_nonzero(catastrophic_translation) == 0
            and np.count_nonzero(catastrophic_rotation) == 0
        ),
        "final_acceptance_reaches_bag_end": bool(final_accepted_to_last_lidar <= 0.25),
        "no_temporal_segment_collapse": bool(minimum_segment_acceptance >= 0.95),
        "all_accepted_poses_finite": bool(np.all(finite_accepted)),
        "short_window_regression_near_identical": short_comparison["near_identical"],
    }
    verdict = "PASS" if all(gate_checks.values()) else "WARN"

    summary = {
        "gate": "Bag D full independent localization stability",
        "verdict": verdict,
        "absolute_accuracy_available": False,
        "absolute_rmse_reported": False,
        "interpretation": "map-relative continuity; not absolute ground-truth accuracy",
        "input_accounting": {
            **full_accounting,
            "first_processed_bag_relative_sec": float(offsets[0]),
            "last_processed_bag_relative_sec": float(offsets[-1]),
        },
        "registration": {
            "accepted_scans": int(np.count_nonzero(accepted_flags)),
            "rejected_scans": int(np.count_nonzero(~accepted_flags)),
            "acceptance_rate": float(np.mean(accepted_flags)),
            "converged_scans": int(np.count_nonzero(converged_flags)),
            "convergence_rate": float(np.mean(converged_flags)),
            "reject_reason_counts": dict(
                sorted(Counter(row["reject_reason"] for row in rejected_rows).items())
            ),
            "rejection_runs": rejection_runs,
            "longest_consecutive_rejection_run_scans": longest_run,
            "inliers": distribution(inliers),
            "final_error_per_inlier": distribution(error_per_inlier),
            "iterations": distribution(iterations),
        },
        "prediction": prediction,
        "gicp_correction": {
            "translation_m": distribution(correction_translation),
            "rotation_deg": distribution(np.degrees(correction_rotation)),
        },
        "accepted_trajectory_continuity": {
            "translation_step_m": distribution(accepted_translation),
            "rotation_step_deg": distribution(np.degrees(accepted_rotation)),
            "catastrophic_translation_threshold_m": 1.0,
            "catastrophic_rotation_threshold_deg": 30.0,
            "contiguous_dt_threshold_sec": 0.25,
            "catastrophic_translation_jump_count": int(
                np.count_nonzero(catastrophic_translation)
            ),
            "catastrophic_rotation_jump_count": int(np.count_nonzero(catastrophic_rotation)),
            "steps_across_rejection_gaps_excluded": int(np.count_nonzero(~contiguous)),
            "path_length_m": float(np.sum(accepted_translation)),
            "finite_accepted_pose_count": int(np.count_nonzero(finite_accepted)),
            "map_xy_bounding_box": {
                "minimum": map_min.tolist(),
                "maximum": map_max.tolist(),
                "accepted_inside_count": int(np.count_nonzero(inside_bbox)),
                "accepted_outside_count": int(np.count_nonzero(~inside_bbox)),
                "inside_fraction": float(np.mean(inside_bbox)),
                "note": "Bounding-box containment plus overlay is a continuity screen, not accuracy proof.",
            },
        },
        "six_dof_behavior": {
            "z_m": distribution(trajectory_positions[:, 2]),
            "roll_deg": distribution(np.degrees(trajectory_rpy[:, 0])),
            "pitch_deg": distribution(np.degrees(trajectory_rpy[:, 1])),
            "z_range_m": float(np.ptp(trajectory_positions[:, 2])),
            "roll_range_deg": float(np.ptp(np.degrees(trajectory_rpy[:, 0]))),
            "pitch_range_deg": float(np.ptp(np.degrees(trajectory_rpy[:, 1]))),
        },
        "runtime": {
            "registration_runtime_ms": distribution(runtime),
            "core_localization_latency_ms": distribution(core_latency),
            "separated": True,
        },
        "time_segments": segments,
        "short_window_regression": short_comparison,
        "gate_checks": gate_checks,
        "minimum_segment_acceptance_rate": minimum_segment_acceptance,
        "final_accepted_to_last_lidar_sec": final_accepted_to_last_lidar,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "short_baseline_comparison.json", short_comparison)

    with (output / "rejection_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id", "start_processed_index", "end_processed_index", "length_scans",
                "start_offset_sec", "end_offset_sec", "duration_sec", "reasons",
            ],
        )
        writer.writeheader()
        for run in rejection_runs:
            writer.writerow({**run, "reasons": json.dumps(run["reasons"], sort_keys=True)})

    with (output / "segment_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "label", "start_offset_sec", "end_offset_sec", "processed_scans",
            "accepted_scans", "rejected_scans", "acceptance_rate",
            "reject_reason_counts", "prediction_translation_max_m",
            "prediction_yaw_max_deg", "gicp_correction_translation_max_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {**segment, "reject_reason_counts": json.dumps(segment["reject_reason_counts"], sort_keys=True)}
            )

    _make_plots(
        output,
        offsets,
        accepted_flags,
        rejected_rows,
        pred_offsets,
        pred_translation,
        pred_yaw,
        correction_translation,
        correction_rotation,
        trajectory_times - args.origin_timestamp,
        trajectory_positions,
        trajectory_rpy,
        map_points,
    )
    _write_report(output / args.report_filename, summary)
    return summary


def _make_plots(output, offsets, accepted_flags, rejected_rows, prediction_offsets,
                prediction_translation, prediction_yaw, correction_translation,
                correction_rotation, accepted_offsets, trajectory_positions,
                trajectory_rpy, map_points):
    figure, axis = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    axis.scatter(
        offsets, accepted_flags.astype(int), s=8,
        c=np.where(accepted_flags, "tab:green", "tab:red"), rasterized=True,
    )
    axis.set(xlabel="bag-relative time [s]", ylabel="accepted", yticks=[0, 1])
    axis.grid(True, alpha=0.3)
    axis.set_title("Full Bag D localization acceptance timeline")
    figure.savefig(output / "acceptance_timeline.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(prediction_offsets, prediction_translation, linewidth=0.9)
    axes[0].axhline(0.5, color="tab:orange", linestyle="--", label="0.5 m screen")
    axes[0].axhline(1.0, color="tab:red", linestyle="--", label="1.0 m screen")
    axes[0].set_ylabel("prediction XY step [m]")
    axes[0].legend()
    axes[1].plot(prediction_offsets, np.degrees(prediction_yaw), linewidth=0.9)
    axes[1].axhline(20.0, color="tab:orange", linestyle="--", label="20 deg screen")
    axes[1].axhline(30.0, color="tab:red", linestyle="--", label="30 deg screen")
    axes[1].set(xlabel="bag-relative time [s]", ylabel="prediction yaw step [deg]")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("EKF planar prediction continuity")
    figure.savefig(output / "prediction_step_over_time.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(offsets, correction_translation, linewidth=0.9)
    axes[0].set_ylabel("translation correction [m]")
    axes[1].plot(offsets, np.degrees(correction_rotation), linewidth=0.9)
    axes[1].set(xlabel="bag-relative time [s]", ylabel="SO(3) correction [deg]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Prediction-to-GICP correction")
    figure.savefig(output / "gicp_correction_over_time.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(accepted_offsets, trajectory_positions[:, 2], linewidth=0.9)
    axis.set(xlabel="bag-relative time [s]", ylabel="T_map_lidar z [m]")
    axis.grid(True, alpha=0.3)
    axis.set_title("Accepted map-relative lidar z")
    figure.savefig(output / "z_over_time.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(accepted_offsets, np.degrees(trajectory_rpy[:, 0]), linewidth=0.9)
    axes[0].set_ylabel("roll [deg]")
    axes[1].plot(accepted_offsets, np.degrees(trajectory_rpy[:, 1]), linewidth=0.9)
    axes[1].set(xlabel="bag-relative time [s]", ylabel="pitch [deg]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Accepted map-relative lidar roll/pitch")
    figure.savefig(output / "roll_pitch_over_time.png", dpi=160)
    plt.close(figure)

    if map_points.shape[0] > 250000:
        map_points = map_points[:: int(np.ceil(map_points.shape[0] / 250000))]
    rejected_positions = np.asarray(
        [row_position(row, "gicp") for row in rejected_rows], dtype=float
    )
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    axis.scatter(map_points[:, 0], map_points[:, 1], s=0.7, c="0.78", label="Bag C PLY map")
    axis.plot(
        trajectory_positions[:, 0], trajectory_positions[:, 1], color="tab:green",
        linewidth=1.4, label="accepted T_map_lidar",
    )
    if rejected_positions.size:
        axis.scatter(
            rejected_positions[:, 0], rejected_positions[:, 1], marker="x", s=45,
            color="tab:red", label="rejected GICP result",
        )
    axis.set(xlabel="map x [m]", ylabel="map y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend()
    axis.set_title("Bag C map and full Bag D localization trajectory")
    figure.savefig(output / "map_trajectory_overlay.png", dpi=180)
    plt.close(figure)


def _write_report(path, summary):
    accounting = summary["input_accounting"]
    registration = summary["registration"]
    prediction = summary["prediction"]
    correction = summary["gicp_correction"]
    trajectory = summary["accepted_trajectory_continuity"]
    six_dof = summary["six_dof_behavior"]
    runtime = summary["runtime"]
    comparison = summary["short_window_regression"]
    full_short_timing = comparison["timing"]["full_first_50s"]
    baseline_timing = comparison["timing"]["post_fix_short_baseline"]
    segment_lines = []
    for segment in summary["time_segments"]:
        segment_lines.append(
            f"| {segment['label']} | {segment['processed_scans']} | "
            f"{segment['accepted_scans']} | {100.0 * segment['acceptance_rate']:.3f}% | "
            f"`{segment['reject_reason_counts']}` | "
            f"{segment['prediction_translation_max_m']:.4f} m / "
            f"{segment['prediction_yaw_max_deg']:.3f} deg | "
            f"{segment['gicp_correction_translation_max_m']:.4f} m |"
        )
    reject_lines = [
        f"| {run['run_id']} | {run['length_scans']} | {run['start_offset_sec']:.6f} | "
        f"{run['end_offset_sec']:.6f} | `{run['reasons']}` |"
        for run in registration["rejection_runs"]
    ] or ["| - | 0 | - | - | `{}` |"]
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in summary["gate_checks"].items()
    )
    text = f"""# Bag D full independent localization stability Gate

## Verdict: {summary['verdict']}

This is a map-relative continuity evaluation of independent Bag D scans against the Bag C PLY map.
Bag D has no independent reference trajectory, and the Bag C map has known vertical deformation;
therefore this report does not claim absolute x/y/z/yaw accuracy or compute RMSE.

{checks}

## Input accounting

- Metadata / received / processed / skipped LiDAR scans: {accounting['bag_metadata_lidar_inputs']} / {accounting['total_lidar_inputs']} / {accounting['processed_scans']} / {accounting['skipped_scans']}
- EOF received / process survived / all inputs accounted: {accounting['bag_eof_received']} / {accounting['process_survived_to_bag_eof']} / {accounting['all_lidar_inputs_accounted']}
- First / last processed time: {accounting['first_processed_bag_relative_sec']:.6f} / {accounting['last_processed_bag_relative_sec']:.6f} s
- Final accepted time: {accounting['final_accepted_bag_relative_sec']:.6f} s ({summary['final_accepted_to_last_lidar_sec']:.6f} s before the last LiDAR input)
- LiDAR gap > {accounting['first_timing_gap']['threshold_sec']:.2f} s detected: {accounting['first_timing_gap']['detected']}

## Registration and rejection continuity

- Accepted / rejected: {registration['accepted_scans']} / {registration['rejected_scans']} ({100.0 * registration['acceptance_rate']:.3f}% accepted)
- Converged: {registration['converged_scans']} ({100.0 * registration['convergence_rate']:.3f}%)
- Reject histogram: `{registration['reject_reason_counts']}`
- Longest consecutive rejection run: {registration['longest_consecutive_rejection_run_scans']} scan(s)
- Inliers mean / p95 / min: {registration['inliers']['mean']:.1f} / {registration['inliers']['p95']:.1f} / {registration['inliers']['min']:.0f}
- Final error/inlier mean / p95 / max: {registration['final_error_per_inlier']['mean']:.6f} / {registration['final_error_per_inlier']['p95']:.6f} / {registration['final_error_per_inlier']['max']:.6f}
- Iterations mean / p95 / max: {registration['iterations']['mean']:.3f} / {registration['iterations']['p95']:.3f} / {registration['iterations']['max']:.0f}

| run | scans | start [s] | end [s] | reasons |
|---:|---:|---:|---:|---|
{chr(10).join(reject_lines)}

## Prediction and GICP correction

- Prediction translation mean / p95 / p99 / max: {_metric_line(prediction['translation_step_m'], 'm')}
- Prediction yaw mean / p95 / p99 / max: {_metric_line(prediction['yaw_step_deg'], 'deg')}
- Prediction steps >0.5 m / >1.0 m: {prediction['over_0_5m_count']} / {prediction['over_1_0m_count']}
- Prediction steps >20 deg / >30 deg: {prediction['over_20deg_count']} / {prediction['over_30deg_count']}
- Previous 9.213 s startup discontinuity recurrence: {prediction['startup_discontinuity_screen']['recurrence_detected']}
- GICP translation correction mean / p95 / p99 / max: {_metric_line(correction['translation_m'], 'm')}
- GICP SO(3) correction mean / p95 / p99 / max: {_metric_line(correction['rotation_deg'], 'deg')}

## Accepted trajectory and 6DoF behavior

- Translation step mean / p95 / p99 / max: {_metric_line(trajectory['translation_step_m'], 'm')}
- Rotation step mean / p95 / p99 / max: {_metric_line(trajectory['rotation_step_deg'], 'deg')}
- Catastrophic contiguous accepted jumps (>1 m / >30 deg): {trajectory['catastrophic_translation_jump_count']} / {trajectory['catastrophic_rotation_jump_count']}
- Accepted poses inside map XY bounding box: {trajectory['map_xy_bounding_box']['accepted_inside_count']} / {registration['accepted_scans']}
- z min / max / range: {six_dof['z_m']['min']:.6f} / {six_dof['z_m']['max']:.6f} / {six_dof['z_range_m']:.6f} m
- roll min / max / range: {six_dof['roll_deg']['min']:.6f} / {six_dof['roll_deg']['max']:.6f} / {six_dof['roll_range_deg']:.6f} deg
- pitch min / max / range: {six_dof['pitch_deg']['min']:.6f} / {six_dof['pitch_deg']['max']:.6f} / {six_dof['pitch_range_deg']:.6f} deg

The map bounding-box check and overlay screen for loss of map-frame continuity; they are not an
absolute localization-accuracy measurement. The vertical values are likewise map-relative only.

## Runtime

- Registration runtime mean / p95 / p99 / max: {_metric_line(runtime['registration_runtime_ms'], 'ms')}
- Core localization latency mean / p95 / p99 / max: {_metric_line(runtime['core_localization_latency_ms'], 'ms')}

The two timings are reported separately: registration runtime covers small_gicp, while core latency
covers the localizer's per-scan processing path.

## Time segments

Prediction maxima are assigned to the segment containing the current scan of each scan-to-scan step.

| interval | processed | accepted | acceptance | rejects | prediction max | correction max |
|---|---:|---:|---:|---|---|---:|
{chr(10).join(segment_lines)}

## 0-50 s replay regression

- Full replay first 50 s accepted / rejected: {comparison['full_first_50s']['accepted_scans']} / {comparison['full_first_50s']['rejected_scans']}
- Saved post-fix short replay accepted / rejected: {comparison['post_fix_short_baseline']['accepted_scans']} / {comparison['post_fix_short_baseline']['rejected_scans']}
- Timestamp/status mismatches: {comparison['status_mismatches']}
- Accepted pose difference p95 / max: {comparison['accepted_pose_translation_difference_m']['p95']:.6f} / {comparison['accepted_pose_translation_difference_m']['max']:.6f} m
- Accepted rotation difference p95 / max: {comparison['accepted_pose_rotation_difference_deg']['p95']:.6f} / {comparison['accepted_pose_rotation_difference_deg']['max']:.6f} deg
- Counts/reasons identical: {comparison['counts_and_reasons_identical']}
- Near-identical replay result: {comparison['near_identical']}
- Full first-50 s registration runtime mean / p95 / max: {full_short_timing['registration_runtime_ms']['mean']:.3f} / {full_short_timing['registration_runtime_ms']['p95']:.3f} / {full_short_timing['registration_runtime_ms']['max']:.3f} ms
- Saved short registration runtime mean / p95 / max: {baseline_timing['registration_runtime_ms']['mean']:.3f} / {baseline_timing['registration_runtime_ms']['p95']:.3f} / {baseline_timing['registration_runtime_ms']['max']:.3f} ms
- Full first-50 s core latency mean / p95 / max: {full_short_timing['core_localization_latency_ms']['mean']:.3f} / {full_short_timing['core_localization_latency_ms']['p95']:.3f} / {full_short_timing['core_localization_latency_ms']['max']:.3f} ms
- Saved short core latency mean / p95 / max: {baseline_timing['core_localization_latency_ms']['mean']:.3f} / {baseline_timing['core_localization_latency_ms']['p95']:.3f} / {baseline_timing['core_localization_latency_ms']['max']:.3f} ms

## Artifacts

- `localization.csv`, `estimated_traj_lidar.tum`, `full_bag_summary.json`
- `summary.json`, `short_baseline_comparison.json`, `segment_summary.csv`, `rejection_runs.csv`
- `acceptance_timeline.png`, `map_trajectory_overlay.png`
- `prediction_step_over_time.png`, `gicp_correction_over_time.png`
- `z_over_time.png`, `roll_pitch_over_time.png`
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--latency-csv", required=True)
    parser.add_argument("--estimated-trajectory", required=True)
    parser.add_argument("--full-bag-summary", required=True)
    parser.add_argument("--short-localization-csv", required=True)
    parser.add_argument("--short-latency-csv", required=True)
    parser.add_argument("--map-ply", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--origin-timestamp", type=float, required=True)
    parser.add_argument("--metadata-duration-sec", type=float, required=True)
    parser.add_argument("--expected-scans", type=int, required=True)
    parser.add_argument("--report-filename", default="full_bag_localization_report.md")
    args = parser.parse_args()
    if args.metadata_duration_sec <= 200.0:
        parser.error("metadata duration must cover the 200 s to EOF segment")
    if args.expected_scans <= 0:
        parser.error("expected scans must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
