#!/usr/bin/env python3

"""Evaluate independent-drive localization without inventing absolute ground-truth error."""

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

from generate_report import (
    quaternion_yaw,
    read_binary_ply_xyz,
    read_localization_csv,
    read_tum,
    save_plot,
)


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
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def row_vector(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in "xyz"], dtype=float)


def row_quaternion(row, prefix):
    quaternion = np.asarray(
        [float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"], dtype=float
    )
    if not np.all(np.isfinite(quaternion)):
        return quaternion
    norm = np.linalg.norm(quaternion)
    return quaternion / norm if norm > 1.0e-12 else np.full(4, np.nan)


def quaternion_step_angles(quaternions):
    if len(quaternions) < 2:
        return np.empty(0)
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def write_location_csv(path, rows, origin_timestamp):
    fieldnames = [
        "timestamp",
        "bag_offset_sec",
        "location_source",
        "x",
        "y",
        "z",
        "yaw_rad",
        "pred_x",
        "pred_y",
        "pred_z",
        "pred_yaw_rad",
        "gicp_x",
        "gicp_y",
        "gicp_z",
        "gicp_yaw_rad",
        "converged",
        "accepted",
        "reject_reason",
        "iterations",
        "num_inliers",
        "final_error",
        "final_error_per_inlier",
        "runtime_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pred_position = row_vector(row, "pred")
            pred_quaternion = row_quaternion(row, "pred")
            gicp_position = row_vector(row, "gicp")
            gicp_quaternion = row_quaternion(row, "gicp")
            gicp_finite = np.all(np.isfinite(gicp_position)) and np.all(
                np.isfinite(gicp_quaternion)
            )
            location_source = "gicp" if gicp_finite else "prediction"
            position = gicp_position if gicp_finite else pred_position
            quaternion = gicp_quaternion if gicp_finite else pred_quaternion
            yaw = (
                float(quaternion_yaw(quaternion))
                if np.all(np.isfinite(quaternion))
                else float("nan")
            )
            pred_yaw = (
                float(quaternion_yaw(pred_quaternion))
                if np.all(np.isfinite(pred_quaternion))
                else float("nan")
            )
            gicp_yaw = (
                float(quaternion_yaw(gicp_quaternion))
                if gicp_finite
                else float("nan")
            )
            inliers = int(row["num_inliers"])
            final_error = float(row["final_error"])
            writer.writerow(
                {
                    "timestamp": row["timestamp"],
                    "bag_offset_sec": float(row["timestamp"]) - origin_timestamp,
                    "location_source": location_source,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "yaw_rad": yaw,
                    "pred_x": pred_position[0],
                    "pred_y": pred_position[1],
                    "pred_z": pred_position[2],
                    "pred_yaw_rad": pred_yaw,
                    "gicp_x": gicp_position[0],
                    "gicp_y": gicp_position[1],
                    "gicp_z": gicp_position[2],
                    "gicp_yaw_rad": gicp_yaw,
                    "converged": row["converged"],
                    "accepted": row["accepted"],
                    "reject_reason": row["reject_reason"],
                    "iterations": row["iterations"],
                    "num_inliers": inliers,
                    "final_error": final_error,
                    "final_error_per_inlier": (
                        final_error / inliers if inliers > 0 else float("nan")
                    ),
                    "runtime_ms": row["runtime_ms"],
                }
            )


def evaluate(args):
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = read_localization_csv(args.localization_csv)
    trajectory_times, trajectory_positions, trajectory_quaternions = read_tum(
        args.estimated_trajectory
    )
    if trajectory_times.size == 0:
        raise ValueError("Independent run produced no accepted trajectory poses")

    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    offsets = timestamps - args.window_origin_timestamp
    epsilon = 1.0e-6
    if np.any(offsets < args.window_start_sec - epsilon) or np.any(
        offsets > args.window_end_sec + epsilon
    ):
        raise ValueError("Localization CSV contains scans outside the requested time window")
    if args.expected_scans > 0 and len(rows) != args.expected_scans:
        raise ValueError(
            f"Expected {args.expected_scans} scans in the time window, got {len(rows)}"
        )

    accepted_flags = np.asarray([row["accepted"] == "1" for row in rows])
    converged_flags = np.asarray([row["converged"] == "1" for row in rows])
    accepted_rows = [row for row in rows if row["accepted"] == "1"]
    rejected_rows = [row for row in rows if row["accepted"] != "1"]
    if len(accepted_rows) != trajectory_times.size:
        raise ValueError("Accepted CSV count does not match the TUM trajectory pose count")
    write_location_csv(
        output_directory / "accepted_locations.csv",
        accepted_rows,
        args.window_origin_timestamp,
    )
    write_location_csv(
        output_directory / "rejected_locations.csv",
        rejected_rows,
        args.window_origin_timestamp,
    )

    runtimes = np.asarray([float(row["runtime_ms"]) for row in rows])
    inliers = np.asarray([float(row["num_inliers"]) for row in rows])
    final_errors = np.asarray([float(row["final_error"]) for row in rows])
    iterations = np.asarray([float(row["iterations"]) for row in rows])
    attempted = np.isfinite(runtimes) & (runtimes > 0.0)
    error_per_inlier = np.divide(
        final_errors,
        inliers,
        out=np.full(final_errors.shape, np.nan),
        where=inliers > 0.0,
    )

    dt = np.diff(trajectory_times)
    if np.any(dt <= 0.0):
        raise ValueError("Accepted trajectory timestamps are not strictly increasing")
    pose_delta = np.diff(trajectory_positions, axis=0)
    translation_steps = np.linalg.norm(pose_delta, axis=1)
    rotation_steps = quaternion_step_angles(trajectory_quaternions)
    yaws = np.unwrap(quaternion_yaw(trajectory_quaternions))
    yaw_steps = np.diff(yaws)
    speeds = translation_steps / dt
    yaw_rates = yaw_steps / dt
    velocity_vectors = pose_delta / dt[:, None]
    acceleration_dt = 0.5 * (dt[1:] + dt[:-1]) if dt.size > 1 else np.empty(0)
    accelerations = (
        np.linalg.norm(np.diff(velocity_vectors, axis=0), axis=1) / acceleration_dt
        if acceleration_dt.size
        else np.empty(0)
    )
    yaw_accelerations = (
        np.abs(np.diff(yaw_rates)) / acceleration_dt
        if acceleration_dt.size
        else np.empty(0)
    )
    step_offsets = trajectory_times[1:] - args.window_origin_timestamp
    largest_translation_index = int(np.argmax(translation_steps)) if translation_steps.size else 0
    largest_rotation_index = int(np.argmax(rotation_steps)) if rotation_steps.size else 0

    with (output_directory / "pose_jumps.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp",
                "bag_offset_sec",
                "dt_sec",
                "translation_step_m",
                "rotation_step_rad",
                "rotation_step_deg",
                "speed_mps",
                "yaw_rate_radps",
            ]
        )
        for index in range(len(dt)):
            writer.writerow(
                [
                    trajectory_times[index + 1],
                    step_offsets[index],
                    dt[index],
                    translation_steps[index],
                    rotation_steps[index],
                    np.degrees(rotation_steps[index]),
                    speeds[index],
                    yaw_rates[index],
                ]
            )

    run_metadata = {}
    run_summary_path = output_directory / "run_summary.json"
    if run_summary_path.exists():
        with run_summary_path.open(encoding="utf-8") as stream:
            run_metadata = json.load(stream)
    summary = {
        "evaluation_name": "independent 163346 map/localization validation",
        "absolute_accuracy_available": False,
        "absolute_rmse_reported": False,
        "reference_trajectory": None,
        "initial_pose_role": "initial guess only; not ground truth",
        "window": {
            "origin_timestamp": args.window_origin_timestamp,
            "start_offset_sec": args.window_start_sec,
            "end_offset_sec": args.window_end_sec,
            "first_processed_offset_sec": float(offsets[0]),
            "last_processed_offset_sec": float(offsets[-1]),
        },
        "total_scans": len(rows),
        "accepted_scans": int(np.count_nonzero(accepted_flags)),
        "rejected_scans": int(np.count_nonzero(~accepted_flags)),
        "acceptance_rate": float(np.mean(accepted_flags)),
        "converged_scans": int(np.count_nonzero(converged_flags)),
        "convergence_rate": float(np.mean(converged_flags)),
        "rejection_reasons": dict(
            sorted(Counter(row["reject_reason"] for row in rejected_rows).items())
        ),
        "registration_attempts": int(np.count_nonzero(attempted)),
        "iterations": distribution(iterations[attempted]),
        "num_inliers": distribution(inliers[attempted]),
        "gicp_final_error": distribution(final_errors[attempted]),
        "gicp_final_error_per_inlier": distribution(error_per_inlier[attempted]),
        "runtime_ms": distribution(runtimes[attempted]),
        "pose_jump": {
            "translation_step_m": distribution(translation_steps),
            "rotation_step_rad": distribution(rotation_steps),
            "rotation_step_deg": distribution(np.degrees(rotation_steps)),
            "largest_translation_step_offset_sec": (
                float(step_offsets[largest_translation_index]) if translation_steps.size else None
            ),
            "largest_rotation_step_offset_sec": (
                float(step_offsets[largest_rotation_index]) if rotation_steps.size else None
            ),
        },
        "trajectory_smoothness": {
            "accepted_poses": int(trajectory_times.size),
            "path_length_m": float(np.sum(translation_steps)),
            "sample_dt_sec": distribution(dt),
            "speed_mps": distribution(speeds),
            "absolute_yaw_rate_radps": distribution(np.abs(yaw_rates)),
            "translational_acceleration_mps2": distribution(accelerations),
            "absolute_yaw_acceleration_radps2": distribution(yaw_accelerations),
        },
        "run_metadata": run_metadata,
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    save_plot(
        output_directory / "acceptance_timeline.png",
        "Independent 163346 accepted/rejected scans (0-50 s)",
        "bag-relative time [s]",
        "accepted (1) / rejected (0)",
        lambda axis: axis.scatter(
            offsets,
            accepted_flags.astype(int),
            c=np.where(accepted_flags, "tab:green", "tab:red"),
            s=12,
        ),
    )
    save_plot(
        output_directory / "num_inliers.png",
        "Independent-run GICP inliers",
        "bag-relative time [s]",
        "number of inliers",
        lambda axis: axis.plot(offsets, inliers),
    )
    save_plot(
        output_directory / "gicp_final_error.png",
        "Independent-run GICP final error",
        "bag-relative time [s]",
        "raw final error",
        lambda axis: axis.plot(offsets, final_errors),
    )
    save_plot(
        output_directory / "runtime_ms.png",
        "Independent-run registration runtime",
        "bag-relative time [s]",
        "runtime [ms]",
        lambda axis: axis.plot(offsets, runtimes),
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(step_offsets, translation_steps)
    axes[0].set_ylabel("translation step [m]")
    axes[1].plot(step_offsets, np.degrees(rotation_steps))
    axes[1].set_ylabel("rotation step [deg]")
    axes[1].set_xlabel("bag-relative time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Accepted-trajectory pose jumps")
    figure.savefig(output_directory / "pose_jumps.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False, constrained_layout=True)
    axes[0].plot(step_offsets, speeds)
    axes[0].set_ylabel("speed [m/s]")
    axes[1].plot(step_offsets, yaw_rates)
    axes[1].set_ylabel("yaw rate [rad/s]")
    if accelerations.size:
        acceleration_offsets = trajectory_times[2:] - args.window_origin_timestamp
        axes[2].plot(acceleration_offsets, accelerations)
    axes[2].set_ylabel("acceleration [m/s²]")
    axes[2].set_xlabel("bag-relative time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Independent accepted-trajectory smoothness")
    figure.savefig(output_directory / "trajectory_smoothness.png", dpi=160)
    plt.close(figure)

    map_points = read_binary_ply_xyz(args.map_ply)
    if map_points.shape[0] > 250000:
        map_points = map_points[:: int(np.ceil(map_points.shape[0] / 250000))]
    rejected_locations = []
    for row in rejected_rows:
        gicp_position = row_vector(row, "gicp")
        rejected_locations.append(
            gicp_position if np.all(np.isfinite(gicp_position)) else row_vector(row, "pred")
        )
    rejected_locations = (
        np.asarray(rejected_locations)
        if rejected_locations
        else np.empty((0, 3), dtype=float)
    )
    save_plot(
        output_directory / "map_trajectory_overlay.png",
        "150626 PLY map with independent 163346 localization",
        "x [m]",
        "y [m]",
        lambda axis: (
            axis.scatter(map_points[:, 0], map_points[:, 1], s=0.5, c="0.8", label="PLY map"),
            axis.plot(
                trajectory_positions[:, 0],
                trajectory_positions[:, 1],
                c="tab:green",
                linewidth=1.2,
                label="accepted T_map_lidar",
            ),
            axis.scatter(
                rejected_locations[:, 0],
                rejected_locations[:, 1],
                c="tab:red",
                marker="x",
                s=24,
                label="rejected",
            ),
        ),
        equal=True,
    )

    runtime = summary["runtime_ms"]
    jumps = summary["pose_jump"]
    smoothness = summary["trajectory_smoothness"]
    report = f"""# Independent 163346 localization validation

This run aligns scans from the separate `163346` drive to the PLY map generated by `150626`.
Only the inclusive bag-relative interval {args.window_start_sec:.1f}-{args.window_end_sec:.1f} s
was processed. No scan after the end bound or after the later sensor gap is part of this report.

## Outcome

- Total scans: {len(rows)}
- Accepted / rejected: {summary['accepted_scans']} / {summary['rejected_scans']}
- Converged: {summary['converged_scans']} ({100.0 * summary['convergence_rate']:.3f}%)
- Inliers mean / p95 / min: {summary['num_inliers']['mean']:.1f} / {summary['num_inliers']['p95']:.1f} / {summary['num_inliers']['min']:.0f}
- Final error mean / p95 / max: {summary['gicp_final_error']['mean']:.3f} / {summary['gicp_final_error']['p95']:.3f} / {summary['gicp_final_error']['max']:.3f}
- Runtime mean / p95 / max: {runtime['mean']:.3f} / {runtime['p95']:.3f} / {runtime['max']:.3f} ms
- Translation step p95 / max: {jumps['translation_step_m']['p95']:.4f} / {jumps['translation_step_m']['max']:.4f} m
- Rotation step p95 / max: {jumps['rotation_step_deg']['p95']:.3f} / {jumps['rotation_step_deg']['max']:.3f} deg
- Accepted path length: {smoothness['path_length_m']:.3f} m
- Rejection reasons: {summary['rejection_reasons']}

## Interpretation boundaries

- This is independent map/localization validation because map and scans come from different
  drives. It is stronger than the same-bag pipeline consistency smoke test for detecting map or
  localization coupling.
- There is no GLIM reference trajectory for `163346`; therefore no absolute position, yaw, or
  trajectory RMSE is claimed or computed.
- The first 150626 map-frame pose is used only as an initial registration seed under the Phase 1
  same-staging-pose assumption. This is not global relocalization and the seed is not ground truth.
- The actual `base_link->velodyne` extrinsic is still unavailable, so the explicit identity
  prediction approximation remains. The evaluated output is `T_map_lidar`.
- Inspect `map_trajectory_overlay.png`, `accepted_locations.csv`, and `rejected_locations.csv`
  together with the numeric jump/smoothness metrics; without an independent reference, they are
  consistency and plausibility evidence rather than absolute accuracy proof.
"""
    (output_directory / "independent_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--estimated-trajectory", required=True)
    parser.add_argument("--map-ply", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--window-origin-timestamp", type=float, required=True)
    parser.add_argument("--window-start-sec", type=float, default=0.0)
    parser.add_argument("--window-end-sec", type=float, default=50.0)
    parser.add_argument("--expected-scans", type=int, default=490)
    args = parser.parse_args()
    if args.window_start_sec < 0.0 or args.window_end_sec < args.window_start_sec:
        parser.error("invalid time window")
    if args.expected_scans < 0:
        parser.error("--expected-scans must be nonnegative")
    evaluate(args)


if __name__ == "__main__":
    main()
