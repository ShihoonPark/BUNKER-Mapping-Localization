#!/usr/bin/env python3

"""Validate the accepted-GICP-anchored planar EKF prediction policy."""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_localization_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POSE_COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No localization rows in {path}")
    return rows


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def pose_values(row, prefix):
    return np.asarray([float(row[f"{prefix}_{name}"]) for name in POSE_COMPONENTS])


def quaternion_norm_error(pose):
    return abs(np.linalg.norm(pose[3:7]) - 1.0)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def evaluate_dataset(rows, origin_timestamp=None, ramp_window=None):
    required_corrections = (
        "correction_translation_m",
        "correction_roll_rad",
        "correction_pitch_rad",
        "correction_yaw_rad",
    )
    missing = [name for name in required_corrections if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing explicit correction columns: {missing}")

    accepted = np.asarray([row["accepted"] == "1" for row in rows])
    converged = np.asarray([row["converged"] == "1" for row in rows])
    attempted = np.asarray([float(row["runtime_ms"]) > 0.0 for row in rows])
    pred = np.asarray([pose_values(row, "pred") for row in rows])
    gicp = np.asarray([pose_values(row, "gicp") for row in rows])
    corrections = np.asarray(
        [[float(row[name]) for name in required_corrections] for row in rows]
    )

    accepted_six_dof = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(gicp), axis=1)
    pred_norm_error = np.asarray([quaternion_norm_error(pose) for pose in pred])
    gicp_norm_error = np.asarray([quaternion_norm_error(pose) for pose in gicp])

    hold_z = []
    hold_roll = []
    hold_pitch = []
    previous_accepted = None
    for index, row in enumerate(rows):
        if previous_accepted is not None:
            hold_z.append(abs(pred[index, 2] - previous_accepted[2]))
            hold_roll.append(abs(wrap(pred[index, 7] - previous_accepted[7])))
            hold_pitch.append(abs(wrap(pred[index, 8] - previous_accepted[8])))
        if row["accepted"] == "1":
            previous_accepted = gicp[index]

    accepted_gicp = gicp[accepted]
    z_steps = np.abs(np.diff(accepted_gicp[:, 2]))
    pitch_steps = np.abs(
        np.diff(np.unwrap(accepted_gicp[:, 8]))
    )
    accepted_corrections = corrections[accepted]
    correction_finite = np.all(np.isfinite(corrections[attempted]), axis=1)

    metrics = {
        "total_scans": len(rows),
        "accepted_scans": int(np.count_nonzero(accepted)),
        "rejected_scans": int(np.count_nonzero(~accepted)),
        "acceptance_rate": float(np.mean(accepted)),
        "converged_scans": int(np.count_nonzero(converged)),
        "rejection_reasons": dict(sorted(Counter(
            row["reject_reason"] for row in rows if row["accepted"] != "1"
        ).items())),
        "accepted_six_dof_finite": int(np.count_nonzero(accepted & accepted_six_dof)),
        "all_accepted_six_dof_finite": bool(np.all(accepted_six_dof[accepted])),
        "prediction_quaternion_norm_error_max": float(np.max(pred_norm_error[accepted])),
        "gicp_quaternion_norm_error_max": float(np.max(gicp_norm_error[accepted])),
        "all_attempted_corrections_finite": bool(np.all(correction_finite)),
        "prediction_hold_error": {
            "z_m": distribution(hold_z),
            "roll_rad": distribution(hold_roll),
            "pitch_rad": distribution(hold_pitch),
        },
        "accepted_correction": {
            "translation_m": distribution(accepted_corrections[:, 0]),
            "absolute_roll_deg": distribution(np.degrees(np.abs(accepted_corrections[:, 1]))),
            "absolute_pitch_deg": distribution(np.degrees(np.abs(accepted_corrections[:, 2]))),
            "absolute_yaw_deg": distribution(np.degrees(np.abs(accepted_corrections[:, 3]))),
        },
        "accepted_gicp_z": distribution(accepted_gicp[:, 2]),
        "accepted_gicp_z_step_m": distribution(z_steps),
        "accepted_gicp_pitch_deg": distribution(np.degrees(accepted_gicp[:, 8])),
        "accepted_gicp_pitch_step_deg": distribution(np.degrees(pitch_steps)),
    }

    times = np.asarray([float(row["timestamp"]) for row in rows])
    offsets = times - origin_timestamp if origin_timestamp is not None else times - times[0]
    metrics["first_offset_sec"] = float(offsets[0])
    metrics["last_offset_sec"] = float(offsets[-1])
    series = {"offsets": offsets, "accepted": accepted, "pred": pred, "gicp": gicp}

    if ramp_window is not None:
        start, end = ramp_window
        ramp = accepted & (offsets >= start) & (offsets <= end)
        ramp_pose = gicp[ramp]
        ramp_z_steps = np.abs(np.diff(ramp_pose[:, 2]))
        ramp_pitch_steps = np.degrees(np.abs(np.diff(np.unwrap(ramp_pose[:, 8]))))
        metrics["ramp"] = {
            "window_sec": [start, end],
            "accepted_scans": int(np.count_nonzero(ramp)),
            "z_range_m": float(np.ptp(ramp_pose[:, 2])),
            "pitch_range_deg": float(np.degrees(np.ptp(ramp_pose[:, 8]))),
            "z_step_m": distribution(ramp_z_steps),
            "pitch_step_deg": distribution(ramp_pitch_steps),
        }
        series["ramp_mask"] = (offsets >= start) & (offsets <= end)
    return metrics, series


def save_ramp_plot(path, title, ylabel, offsets, prediction, registration, accepted):
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(offsets, prediction, label="planar EKF prediction", linewidth=1.2)
    axis.scatter(
        offsets[accepted], registration[accepted], label="accepted full-6DoF GICP",
        s=14, c="tab:green",
    )
    axis.scatter(
        offsets[~accepted], registration[~accepted], label="rejected GICP", s=22,
        c="tab:red", marker="x",
    )
    axis.set_title(title)
    axis.set_xlabel("bag-relative time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def fmt(stats):
    return f"{stats['mean']:.6f} / {stats['p95']:.6f} / {stats['max']:.6f}"


def evaluate(args):
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    same_metrics, _ = evaluate_dataset(read_rows(args.same_bag_csv))
    independent_metrics, independent_series = evaluate_dataset(
        read_rows(args.independent_csv),
        origin_timestamp=args.independent_origin_timestamp,
        ramp_window=(args.ramp_start_sec, args.ramp_end_sec),
    )

    gates = {
        "same_bag_scan_count": same_metrics["total_scans"] == args.expected_same_bag_scans,
        "same_bag_acceptance_regression": same_metrics["accepted_scans"] >= args.minimum_same_bag_accepted,
        "independent_scan_count": independent_metrics["total_scans"] == args.expected_independent_scans,
        "independent_acceptance_regression": independent_metrics["accepted_scans"] >= args.minimum_independent_accepted,
        "independent_window_end": independent_metrics["last_offset_sec"] <= 50.0 + 1.0e-6,
        "accepted_6dof_finite": (
            same_metrics["all_accepted_six_dof_finite"] and
            independent_metrics["all_accepted_six_dof_finite"]
        ),
        "quaternions_normalized": max(
            same_metrics["prediction_quaternion_norm_error_max"],
            same_metrics["gicp_quaternion_norm_error_max"],
            independent_metrics["prediction_quaternion_norm_error_max"],
            independent_metrics["gicp_quaternion_norm_error_max"],
        ) <= 1.0e-6,
        "corrections_finite": (
            same_metrics["all_attempted_corrections_finite"] and
            independent_metrics["all_attempted_corrections_finite"]
        ),
        "ekf_z_roll_pitch_do_not_accumulate": max(
            same_metrics["prediction_hold_error"][axis]["max"]
            for axis in ("z_m", "roll_rad", "pitch_rad")
        ) <= 1.0e-9 and max(
            independent_metrics["prediction_hold_error"][axis]["max"]
            for axis in ("z_m", "roll_rad", "pitch_rad")
        ) <= 1.0e-9,
        "ramp_z_changes_continuously": (
            independent_metrics["ramp"]["z_range_m"] > 0.05 and
            independent_metrics["ramp"]["z_step_m"]["max"] < 1.0
        ),
        "ramp_pitch_changes_continuously": (
            independent_metrics["ramp"]["pitch_range_deg"] > 1.0 and
            independent_metrics["ramp"]["pitch_step_deg"]["max"] < 30.0
        ),
    }
    passed = all(gates.values())
    summary = {
        "gate_name": "accepted GICP 6DoF anchor plus planar EKF relative prediction",
        "production_filter": "robot_localization ekf_node",
        "final_output": "T_map_lidar",
        "map_base_conversion": "T_map_base = T_map_lidar * inverse(T_base_lidar)",
        "acceptance_regression_baseline": {
            "same_bag_accepted": args.baseline_same_bag_accepted,
            "independent_accepted": args.baseline_independent_accepted,
            "minimum_same_bag_accepted": args.minimum_same_bag_accepted,
            "minimum_independent_accepted": args.minimum_independent_accepted,
        },
        "same_bag": same_metrics,
        "independent_0_50s": independent_metrics,
        "success_criteria": gates,
        "passed": passed,
    }
    with (output / "planar_ekf_gate_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    ramp = independent_series["ramp_mask"]
    ramp_offsets = independent_series["offsets"][ramp]
    ramp_accepted = independent_series["accepted"][ramp]
    ramp_pred = independent_series["pred"][ramp]
    ramp_gicp = independent_series["gicp"][ramp]
    save_ramp_plot(
        output / "independent_ramp_z.png",
        "Independent ramp: held prediction z and full-6DoF GICP z",
        "z [m]", ramp_offsets, ramp_pred[:, 2], ramp_gicp[:, 2], ramp_accepted,
    )
    save_ramp_plot(
        output / "independent_ramp_pitch.png",
        "Independent ramp: held prediction pitch and full-6DoF GICP pitch",
        "pitch [deg]", ramp_offsets, np.degrees(ramp_pred[:, 8]),
        np.degrees(ramp_gicp[:, 8]), ramp_accepted,
    )

    same_correction = same_metrics["accepted_correction"]
    independent_correction = independent_metrics["accepted_correction"]
    ramp_metrics = independent_metrics["ramp"]
    report = f"""# Accepted-GICP-anchored planar EKF Gate

The production baseline remains the official `robot_localization/ekf_node`. Its role is only to
provide scan-to-scan `dx`, `dy`, and `dyaw` for the small_gicp initial guess. Every prediction is
anchored at the last accepted full-6DoF `T_map_lidar`: z, roll, and pitch are held from that anchor,
and small_gicp remains responsible for the full-6DoF result on every scan.

## Outcome

| Dataset | scans | accepted / rejected | prior accepted | delta | finite accepted 6DoF | correction translation mean/p95/max [m] |
|---|---:|---:|---:|---:|---:|---:|
| 150626 full | {same_metrics['total_scans']} | {same_metrics['accepted_scans']} / {same_metrics['rejected_scans']} | {args.baseline_same_bag_accepted} | {same_metrics['accepted_scans'] - args.baseline_same_bag_accepted:+d} | {same_metrics['accepted_six_dof_finite']} | {fmt(same_correction['translation_m'])} |
| 163346 0-50 s | {independent_metrics['total_scans']} | {independent_metrics['accepted_scans']} / {independent_metrics['rejected_scans']} | {args.baseline_independent_accepted} | {independent_metrics['accepted_scans'] - args.baseline_independent_accepted:+d} | {independent_metrics['accepted_six_dof_finite']} | {fmt(independent_correction['translation_m'])} |

- Same-bag rejection reasons: {same_metrics['rejection_reasons']}
- Independent rejection reasons: {independent_metrics['rejection_reasons']}
- Maximum prediction hold errors, independent z/roll/pitch: {independent_metrics['prediction_hold_error']['z_m']['max']:.3e} m / {independent_metrics['prediction_hold_error']['roll_rad']['max']:.3e} rad / {independent_metrics['prediction_hold_error']['pitch_rad']['max']:.3e} rad
- Maximum accepted quaternion norm error: {max(independent_metrics['prediction_quaternion_norm_error_max'], independent_metrics['gicp_quaternion_norm_error_max']):.3e}
- Explicit correction roll mean/p95/max: {fmt(independent_correction['absolute_roll_deg'])} deg
- Explicit correction pitch mean/p95/max: {fmt(independent_correction['absolute_pitch_deg'])} deg
- Explicit correction yaw mean/p95/max: {fmt(independent_correction['absolute_yaw_deg'])} deg

## Ramp, {args.ramp_start_sec:.1f}-{args.ramp_end_sec:.1f} s

- Accepted ramp scans: {ramp_metrics['accepted_scans']}
- GICP z range: {ramp_metrics['z_range_m']:.6f} m
- GICP z step mean/p95/max: {fmt(ramp_metrics['z_step_m'])} m
- GICP pitch range: {ramp_metrics['pitch_range_deg']:.6f} deg
- GICP pitch step mean/p95/max: {fmt(ramp_metrics['pitch_step_deg'])} deg

The stair-step prediction curves in `independent_ramp_z.png` and
`independent_ramp_pitch.png` are intentional: each holds the last accepted GICP z/pitch until the
next full-6DoF registration. The accepted GICP curves show the ramp geometry continuously.

The independent acceptance regression allows at most two additional rejects (0.408 percentage
point) relative to 487/490. No retry, GICP parameter change, voxel change, or quality-gate change
is used to meet that tolerance.

## Gate decision

Overall: **{'PASS' if passed else 'FAIL'}**

""" + "\n".join(
        f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in gates.items()
    ) + """

The actual `T_base_lidar` is still unavailable, so no `T_map_base` output is fabricated. The
tested conversion interface uses `T_map_base = T_map_lidar * T_lidar_base`, where
`T_lidar_base = inverse(T_base_lidar)` after measured calibration is supplied.
"""
    (output / "planar_ekf_gate_report.md").write_text(report, encoding="utf-8")
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Planar EKF Gate failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-bag-csv", required=True)
    parser.add_argument("--independent-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--independent-origin-timestamp", type=float, default=1786692827.2210245)
    parser.add_argument("--ramp-start-sec", type=float, default=25.0)
    parser.add_argument("--ramp-end-sec", type=float, default=35.0)
    parser.add_argument("--expected-same-bag-scans", type=int, default=869)
    parser.add_argument("--expected-independent-scans", type=int, default=490)
    parser.add_argument("--minimum-same-bag-accepted", type=int, default=854)
    parser.add_argument("--minimum-independent-accepted", type=int, default=485)
    parser.add_argument("--baseline-same-bag-accepted", type=int, default=854)
    parser.add_argument("--baseline-independent-accepted", type=int, default=487)
    args = parser.parse_args()
    if args.ramp_end_sec <= args.ramp_start_sec:
        parser.error("ramp end must be after ramp start")
    evaluate(args)


if __name__ == "__main__":
    main()
