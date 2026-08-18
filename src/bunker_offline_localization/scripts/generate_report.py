#!/usr/bin/env python3

"""Generate Phase 1 pipeline/reference consistency metrics and diagnostic plots."""

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


def read_tum(path):
    timestamps = []
    positions = []
    quaternions = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
            values = np.asarray([float(field) for field in fields], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}:{line_number}: non-finite TUM value")
            quaternion = values[4:8]
            norm = np.linalg.norm(quaternion)
            if norm < 1.0e-12:
                raise ValueError(f"{path}:{line_number}: zero quaternion")
            timestamps.append(values[0])
            positions.append(values[1:4])
            quaternions.append(quaternion / norm)
    if not timestamps:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 4))
    timestamps = np.asarray(timestamps)
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{path}: timestamps are not sorted")
    return timestamps, np.asarray(positions), np.asarray(quaternions)


def quaternion_yaw(quaternion_xyzw):
    x, y, z, w = np.moveaxis(quaternion_xyzw, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def associate_nearest(query_times, reference_times, tolerance):
    associations = []
    for query_index, timestamp in enumerate(query_times):
        upper = int(np.searchsorted(reference_times, timestamp))
        candidates = []
        if upper < len(reference_times):
            candidates.append(upper)
        if upper > 0:
            candidates.append(upper - 1)
        if not candidates:
            continue
        reference_index = min(
            candidates, key=lambda index: abs(reference_times[index] - timestamp)
        )
        difference = abs(reference_times[reference_index] - timestamp)
        if difference <= tolerance:
            associations.append((query_index, reference_index, difference))
    return associations


def read_localization_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Localization CSV has no records: {path}")
    required = {
        "timestamp",
        "accepted",
        "reject_reason",
        "gicp_x",
        "gicp_y",
        "gicp_z",
        "gicp_qx",
        "gicp_qy",
        "gicp_qz",
        "gicp_qw",
        "final_error",
        "num_inliers",
        "runtime_ms",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Localization CSV is missing fields: {sorted(missing)}")
    return rows


def read_binary_ply_xyz(path):
    scalar_types = {
        "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
        "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
        "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    with Path(path).open("rb") as stream:
        if stream.readline().decode("ascii").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        vertex_properties = []
        active_element = None
        little_endian = False
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw_line.decode("ascii").strip()
            fields = line.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                little_endian = True
            elif fields and fields[0] == "element":
                active_element = fields[1]
                if active_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields and fields[0] == "property" and active_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                if fields[1] not in scalar_types:
                    raise ValueError(f"Unsupported PLY scalar type: {fields[1]}")
                vertex_properties.append((fields[2], scalar_types[fields[1]]))
            elif line == "end_header":
                break
        if not little_endian or vertex_count is None:
            raise ValueError("Expected binary_little_endian PLY vertices")
        names = [name for name, _ in vertex_properties]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError("PLY vertex does not contain x/y/z")
        dtype = np.dtype([(name, "<" + code) for name, code in vertex_properties])
        vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            raise ValueError("PLY vertex payload is truncated")
        return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "rmse": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "max": float(np.max(values)),
    }


def save_plot(path, title, xlabel, ylabel, plotter, equal=False):
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plotter(axis)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if equal:
        axis.set_aspect("equal", adjustable="datalim")
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate(args):
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = read_localization_csv(args.localization_csv)
    estimated_times, estimated_positions, estimated_quaternions = read_tum(
        args.estimated_trajectory
    )
    reference_times, reference_positions, reference_quaternions = read_tum(
        args.reference_trajectory
    )
    if estimated_times.size == 0:
        raise ValueError("No accepted estimated poses are available for comparison")

    associations = associate_nearest(
        estimated_times, reference_times, args.timestamp_tolerance
    )
    if not associations:
        raise ValueError("No estimated/reference timestamps satisfy the tolerance")
    estimated_indices = np.asarray([item[0] for item in associations], dtype=int)
    reference_indices = np.asarray([item[1] for item in associations], dtype=int)
    time_differences = np.asarray([item[2] for item in associations])
    estimate_position = estimated_positions[estimated_indices]
    reference_position = reference_positions[reference_indices]
    position_error = estimate_position - reference_position
    translation_error = np.linalg.norm(position_error, axis=1)

    estimate_quaternion = estimated_quaternions[estimated_indices]
    reference_quaternion = reference_quaternions[reference_indices]
    quaternion_dot = np.abs(np.sum(estimate_quaternion * reference_quaternion, axis=1))
    rotation_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    estimate_yaw = quaternion_yaw(estimate_quaternion)
    reference_yaw = quaternion_yaw(reference_quaternion)
    yaw_error = np.arctan2(
        np.sin(estimate_yaw - reference_yaw), np.cos(estimate_yaw - reference_yaw)
    )

    with (output_directory / "reference_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp", "reference_timestamp", "timestamp_difference",
            "estimated_x", "estimated_y", "estimated_z",
            "reference_x", "reference_y", "reference_z",
            "error_x", "error_y", "error_z", "translation_error",
            "estimated_yaw", "reference_yaw", "yaw_error", "rotation_angle_error",
        ])
        for index in range(len(associations)):
            writer.writerow([
                estimated_times[estimated_indices[index]],
                reference_times[reference_indices[index]], time_differences[index],
                *estimate_position[index], *reference_position[index], *position_error[index],
                translation_error[index], estimate_yaw[index], reference_yaw[index],
                yaw_error[index], rotation_error[index],
            ])

    accepted_count = sum(row["accepted"] == "1" for row in rows)
    rejected_count = len(rows) - accepted_count
    rejection_reasons = Counter(
        row["reject_reason"] for row in rows if row["accepted"] != "1"
    )
    runtimes = np.asarray([float(row["runtime_ms"]) for row in rows])
    final_errors = np.asarray([float(row["final_error"]) for row in rows])
    num_inliers = np.asarray([float(row["num_inliers"]) for row in rows])
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    accepted_flags = np.asarray([row["accepted"] == "1" for row in rows])
    registered = np.isfinite(runtimes) & (runtimes > 0.0)
    runtime_finite = runtimes[registered]
    runtime_summary = finite_stats(runtime_finite)
    runtime_summary["p95"] = (
        float(np.percentile(runtime_finite, 95)) if runtime_finite.size else None
    )
    runtime_summary["total"] = (
        float(np.sum(runtime_finite)) if runtime_finite.size else None
    )

    summary = {
        "evaluation_name": "pipeline/reference consistency smoke test",
        "independent_localization_accuracy": False,
        "timestamp_tolerance_seconds": args.timestamp_tolerance,
        "processed_scans": len(rows),
        "accepted_scans": accepted_count,
        "rejected_scans": rejected_count,
        "acceptance_rate": accepted_count / len(rows),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "accepted_trajectory_poses": int(estimated_times.size),
        "associated_reference_poses": len(associations),
        "timestamp_difference_seconds": finite_stats(time_differences),
        "translation_error_m": finite_stats(translation_error),
        "component_rmse_m": {
            "x": float(np.sqrt(np.mean(position_error[:, 0] ** 2))),
            "y": float(np.sqrt(np.mean(position_error[:, 1] ** 2))),
            "z": float(np.sqrt(np.mean(position_error[:, 2] ** 2))),
        },
        "yaw_error_rad": finite_stats(np.abs(yaw_error)),
        "yaw_error_deg": finite_stats(np.degrees(np.abs(yaw_error))),
        "rotation_angle_error_rad": finite_stats(rotation_error),
        "runtime_ms": runtime_summary,
        "gicp_final_error": finite_stats(final_errors[registered]),
        "num_inliers": finite_stats(num_inliers[registered]),
    }
    run_summary_path = output_directory / "run_summary.json"
    if run_summary_path.exists():
        with run_summary_path.open(encoding="utf-8") as stream:
            summary["run_metadata"] = json.load(stream)
    with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    relative_estimate_time = estimated_times - reference_times[0]
    relative_reference_time = reference_times - reference_times[0]
    relative_comparison_time = estimated_times[estimated_indices] - reference_times[0]
    relative_registration_time = timestamps - reference_times[0]
    save_plot(
        output_directory / "trajectory_xy.png", "Estimated vs GLIM reference trajectory (XY)",
        "x [m]", "y [m]",
        lambda axis: (
            axis.plot(reference_positions[:, 0], reference_positions[:, 1], label="GLIM reference"),
            axis.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="estimated"),
        ), equal=True,
    )
    save_plot(
        output_directory / "z_over_time.png", "Z over time",
        "time from first reference [s]", "z [m]",
        lambda axis: (
            axis.plot(relative_reference_time, reference_positions[:, 2], label="GLIM reference"),
            axis.plot(relative_estimate_time, estimated_positions[:, 2], label="estimated"),
        ),
    )
    save_plot(
        output_directory / "yaw_over_time.png", "Yaw over time",
        "time from first reference [s]", "unwrapped yaw [rad]",
        lambda axis: (
            axis.plot(relative_reference_time, np.unwrap(quaternion_yaw(reference_quaternions)), label="GLIM reference"),
            axis.plot(relative_estimate_time, np.unwrap(quaternion_yaw(estimated_quaternions)), label="estimated"),
        ),
    )
    save_plot(
        output_directory / "translation_error.png", "Translation error vs GLIM reference",
        "time from first reference [s]", "translation error [m]",
        lambda axis: axis.plot(relative_comparison_time, translation_error),
    )
    save_plot(
        output_directory / "yaw_error.png", "Yaw error vs GLIM reference",
        "time from first reference [s]", "yaw error [deg]",
        lambda axis: axis.plot(relative_comparison_time, np.degrees(yaw_error)),
    )
    save_plot(
        output_directory / "gicp_final_error.png", "GICP final error",
        "time from first reference [s]", "raw final error",
        lambda axis: axis.plot(relative_registration_time, final_errors),
    )
    save_plot(
        output_directory / "num_inliers.png", "GICP inliers",
        "time from first reference [s]", "number of inliers",
        lambda axis: axis.plot(relative_registration_time, num_inliers),
    )
    save_plot(
        output_directory / "runtime_ms.png", "Registration runtime",
        "time from first reference [s]", "runtime [ms]",
        lambda axis: axis.plot(relative_registration_time, runtimes),
    )
    save_plot(
        output_directory / "acceptance_timeline.png", "Accepted/rejected scan timeline",
        "time from first reference [s]", "accepted (1) / rejected (0)",
        lambda axis: axis.scatter(
            relative_registration_time, accepted_flags.astype(int),
            c=np.where(accepted_flags, "tab:green", "tab:red"), s=10,
        ),
    )
    map_points = read_binary_ply_xyz(args.map_ply)
    save_plot(
        output_directory / "map_trajectory_overlay.png",
        "Global PLY map with estimated/reference trajectories", "x [m]", "y [m]",
        lambda axis: (
            axis.scatter(map_points[:, 0], map_points[:, 1], s=1, c="0.75", label="PLY map"),
            axis.plot(reference_positions[:, 0], reference_positions[:, 1], label="GLIM reference"),
            axis.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="estimated"),
        ), equal=True,
    )

    identity_approximation = summary.get("run_metadata", {}).get(
        "prediction_uses_identity_base_to_lidar_approximation", True
    )
    translation = summary["translation_error_m"]
    yaw_degrees = summary["yaw_error_deg"]
    runtime = summary["runtime_ms"]
    diagnosis = f"""# Phase 1 localization diagnosis

This is a **pipeline/reference consistency smoke test**, not an independent localization
accuracy measurement. The tested PLY map and GLIM reference were generated from the same bag.

## Run outcome

- Processed scans: {len(rows)}
- Accepted: {accepted_count}
- Rejected: {rejected_count}
- Reference-associated accepted poses: {len(associations)}
- Translation error RMSE: {translation['rmse']:.6f} m
- Translation error mean/max: {translation['mean']:.6f} / {translation['max']:.6f} m
- Yaw absolute error RMSE/max: {yaw_degrees['rmse']:.6f} / {yaw_degrees['max']:.6f} deg
- Registration runtime mean/p95/max: {runtime['mean']:.3f} / {runtime['p95']:.3f} / {runtime['max']:.3f} ms
- Rejection reasons: {dict(sorted(rejection_reasons.items()))}

## Interpretation and limitations

- `target=global PLY map`, `source=current LiDAR scan`, and the recorded result is
  `T_map_lidar` such that `p_map = T_map_lidar * p_lidar`.
- IMU orientation was not fused because every bag quaternion is fixed identity.
- Source covariances are all zero; the adapter used documented initial-tuning assumptions on
  separate topics without changing the bag.
- `two_d_mode` is false. EKF provides a motion initial guess from planar odometry and yaw-rate;
  small_gicp performs the full 6DoF correction against the map.
- Identity `base_link->velodyne` prediction approximation used: {identity_approximation}.
  Until the measured extrinsic is supplied, base_link localization and map->odom validation are
  incomplete even though `T_map_lidar` can be checked in this Phase 1 test.
"""
    (output_directory / "diagnosis.md").write_text(diagnosis, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--estimated-trajectory", required=True)
    parser.add_argument("--reference-trajectory", required=True)
    parser.add_argument("--map-ply", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.06)
    args = parser.parse_args()
    if args.timestamp_tolerance <= 0.0:
        parser.error("--timestamp-tolerance must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
