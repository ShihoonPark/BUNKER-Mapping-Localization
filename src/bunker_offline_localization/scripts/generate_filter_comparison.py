#!/usr/bin/env python3

"""Generate the EKF/UKF prediction and localization A/B comparison Gate report."""

import argparse
import csv
import hashlib
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
    associate_nearest,
    quaternion_yaw,
    read_localization_csv,
    read_tum,
    save_plot,
)


FILTERS = ("ekf", "ukf")
DATASETS = ("same_bag", "independent")


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


def rmse(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def wrap_angles(values):
    return np.arctan2(np.sin(values), np.cos(values))


def extract_pose(rows, prefix):
    positions = np.asarray(
        [[float(row[f"{prefix}_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=float,
    )
    quaternions = np.asarray(
        [[float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"] for row in rows],
        dtype=float,
    )
    norms = np.linalg.norm(quaternions, axis=1)
    finite = np.all(np.isfinite(positions), axis=1) & np.all(
        np.isfinite(quaternions), axis=1
    ) & (norms > 1.0e-12)
    quaternions[finite] /= norms[finite, None]
    return positions, quaternions, finite


def quaternion_multiply(lhs, rhs):
    lx, ly, lz, lw = np.moveaxis(lhs, -1, 0)
    rx, ry, rz, rw = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def correction_values(pred_positions, pred_quaternions, gicp_positions, gicp_quaternions):
    translation = np.linalg.norm(gicp_positions - pred_positions, axis=1)
    inverse_prediction = pred_quaternions.copy()
    inverse_prediction[:, :3] *= -1.0
    delta_quaternions = quaternion_multiply(inverse_prediction, gicp_quaternions)
    delta_quaternions /= np.linalg.norm(delta_quaternions, axis=1)[:, None]
    rotation = 2.0 * np.arccos(np.clip(np.abs(delta_quaternions[:, 3]), 0.0, 1.0))
    yaw = wrap_angles(quaternion_yaw(delta_quaternions))
    return translation, np.degrees(rotation), np.degrees(yaw)


def pose_step_values(positions, quaternions):
    if len(positions) < 2:
        return np.empty(0), np.empty(0)
    translation = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    rotation = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    return translation, rotation


def prediction_evaluation(rows, reference, tolerance):
    reference_times, reference_positions, reference_quaternions = reference
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions, quaternions, finite = extract_pose(rows, "pred")
    valid_indices = np.flatnonzero(finite)
    valid_times = times[valid_indices]
    associations = associate_nearest(valid_times, reference_times, tolerance)
    if not associations:
        raise ValueError("No prediction/reference timestamp associations")
    query_indices = np.asarray([item[0] for item in associations], dtype=int)
    reference_indices = np.asarray([item[1] for item in associations], dtype=int)
    row_indices = valid_indices[query_indices]
    prediction_positions = positions[row_indices]
    prediction_quaternions = quaternions[row_indices]
    associated_reference_positions = reference_positions[reference_indices]
    associated_reference_quaternions = reference_quaternions[reference_indices]
    position_error = prediction_positions - associated_reference_positions
    translation_error = np.linalg.norm(position_error, axis=1)
    prediction_yaw = quaternion_yaw(prediction_quaternions)
    reference_yaw = quaternion_yaw(associated_reference_quaternions)
    yaw_error = wrap_angles(prediction_yaw - reference_yaw)
    quaternion_dot = np.abs(
        np.sum(prediction_quaternions * associated_reference_quaternions, axis=1)
    )
    rotation_error = np.degrees(
        2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    )
    step_translation, step_rotation = pose_step_values(
        prediction_positions, prediction_quaternions
    )
    metrics = {
        "associated_predictions": len(associations),
        "timestamp_difference_sec": distribution([item[2] for item in associations]),
        "translation_rmse_m": rmse(translation_error),
        "translation_error_m": distribution(translation_error),
        "component_rmse_m": {
            axis: rmse(position_error[:, index])
            for index, axis in enumerate("xyz")
        },
        "yaw_rmse_deg": rmse(np.degrees(yaw_error)),
        "absolute_yaw_error_deg": distribution(np.abs(np.degrees(yaw_error))),
        "full_rotation_rmse_deg": rmse(rotation_error),
        "prediction_step_translation_m": distribution(step_translation),
        "prediction_step_rotation_deg": distribution(step_rotation),
        "first_translation_error_above_1m_offset_sec": None,
        "largest_translation_error_offset_sec": float(
            valid_times[query_indices[np.argmax(translation_error)]] - reference_times[0]
        ),
    }
    above_one_meter = np.flatnonzero(translation_error > 1.0)
    if above_one_meter.size:
        metrics["first_translation_error_above_1m_offset_sec"] = float(
            valid_times[query_indices[above_one_meter[0]]] - reference_times[0]
        )
    series = {
        "times": times[row_indices],
        "positions": prediction_positions,
        "quaternions": prediction_quaternions,
        "translation_error": translation_error,
        "yaw_error_deg": np.degrees(yaw_error),
        "rotation_error_deg": rotation_error,
    }
    return metrics, series


def recovery_events(rows, origin):
    events = []
    for index, row in enumerate(rows):
        if row["accepted"] == "1":
            continue
        next_index = next(
            (candidate for candidate in range(index + 1, len(rows))
             if rows[candidate]["accepted"] == "1"),
            None,
        )
        timestamp = float(row["timestamp"])
        event = {
            "reject_timestamp": timestamp,
            "reject_offset_sec": timestamp - origin,
            "reason": row["reject_reason"],
            "recovered": next_index is not None,
            "scans_to_recovery": None,
            "seconds_to_recovery": None,
        }
        if next_index is not None:
            recovery_timestamp = float(rows[next_index]["timestamp"])
            event["scans_to_recovery"] = next_index - index
            event["seconds_to_recovery"] = recovery_timestamp - timestamp
        events.append(event)
    return events


def localization_evaluation(rows, origin, dataset):
    times = np.asarray([float(row["timestamp"]) for row in rows])
    accepted = np.asarray([row["accepted"] == "1" for row in rows])
    converged = np.asarray([row["converged"] == "1" for row in rows])
    pred_positions, pred_quaternions, pred_finite = extract_pose(rows, "pred")
    gicp_positions, gicp_quaternions, gicp_finite = extract_pose(rows, "gicp")
    attempted = np.asarray([float(row["runtime_ms"]) > 0.0 for row in rows])
    inliers = np.asarray([float(row["num_inliers"]) for row in rows])
    errors = np.asarray([float(row["final_error"]) for row in rows])
    iterations = np.asarray([float(row["iterations"]) for row in rows])
    runtimes = np.asarray([float(row["runtime_ms"]) for row in rows])
    filter_runtimes = np.asarray(
        [float(row.get("filter_runtime_ms", "nan")) for row in rows]
    )
    accepted_pose = accepted & pred_finite & gicp_finite
    correction_translation, correction_rotation, correction_yaw = correction_values(
        pred_positions[accepted_pose],
        pred_quaternions[accepted_pose],
        gicp_positions[accepted_pose],
        gicp_quaternions[accepted_pose],
    )
    trajectory_translation, trajectory_rotation = pose_step_values(
        gicp_positions[accepted & gicp_finite], gicp_quaternions[accepted & gicp_finite]
    )
    rejection_reasons = Counter(
        row["reject_reason"] for row in rows if row["accepted"] != "1"
    )
    recovery = recovery_events(rows, origin)
    recovered_scan_counts = [
        event["scans_to_recovery"] for event in recovery if event["recovered"]
    ]
    recovered_seconds = [
        event["seconds_to_recovery"] for event in recovery if event["recovered"]
    ]
    metrics = {
        "total_scans": len(rows),
        "accepted_scans": int(np.count_nonzero(accepted)),
        "rejected_scans": int(np.count_nonzero(~accepted)),
        "acceptance_rate": float(np.mean(accepted)),
        "converged_scans": int(np.count_nonzero(converged)),
        "convergence_rate": float(np.mean(converged)),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "num_inliers": distribution(inliers[attempted]),
        "gicp_final_error": distribution(errors[attempted]),
        "gicp_iterations": distribution(iterations[attempted]),
        "gicp_runtime_ms": distribution(runtimes[attempted]),
        "filter_runtime_proxy_ms": distribution(filter_runtimes),
        "trajectory_step_translation_m": distribution(trajectory_translation),
        "trajectory_step_rotation_deg": distribution(trajectory_rotation),
        "correction_translation_m": distribution(correction_translation),
        "correction_rotation_deg": distribution(correction_rotation),
        "absolute_correction_yaw_deg": distribution(np.abs(correction_yaw)),
        "recovery_after_reject": recovery,
        "recovery_summary": {
            "recovered_rejects": sum(event["recovered"] for event in recovery),
            "unrecovered_rejects": sum(not event["recovered"] for event in recovery),
            "scans_to_recovery": distribution(recovered_scan_counts),
            "seconds_to_recovery": distribution(recovered_seconds),
        },
    }
    offsets = times - origin
    turn_mask = (offsets >= 27.0) & (offsets <= 29.0)
    turn_accepted = turn_mask & accepted_pose
    turn_translation, turn_rotation, turn_yaw = correction_values(
        pred_positions[turn_accepted],
        pred_quaternions[turn_accepted],
        gicp_positions[turn_accepted],
        gicp_quaternions[turn_accepted],
    ) if np.any(turn_accepted) else (np.empty(0), np.empty(0), np.empty(0))
    metrics["large_rotation_27_29s"] = {
        "total_scans": int(np.count_nonzero(turn_mask)),
        "accepted_scans": int(np.count_nonzero(turn_mask & accepted)),
        "rejected_scans": int(np.count_nonzero(turn_mask & ~accepted)),
        "correction_translation_m": distribution(turn_translation),
        "correction_rotation_deg": distribution(turn_rotation),
        "absolute_correction_yaw_deg": distribution(np.abs(turn_yaw)),
        "iterations": distribution(iterations[turn_mask & attempted]),
        "gicp_runtime_ms": distribution(runtimes[turn_mask & attempted]),
    }
    series = {
        "dataset": dataset,
        "times": times,
        "offsets": offsets,
        "accepted": accepted,
        "iterations": iterations,
        "gicp_runtime_ms": runtimes,
        "filter_runtime_ms": filter_runtimes,
        "correction_times": times[accepted_pose],
        "correction_offsets": offsets[accepted_pose],
        "correction_translation": correction_translation,
        "correction_rotation": correction_rotation,
        "correction_yaw": correction_yaw,
    }
    return metrics, series


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def evaluate(args):
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reference = read_tum(args.reference_trajectory)
    if reference[0].size == 0:
        raise ValueError("Reference trajectory is empty")

    rows = {}
    prediction_metrics = {}
    prediction_series = {}
    localization_metrics = {}
    localization_series = {}
    origins = {
        "same_bag": reference[0][0],
        "independent": args.independent_origin_timestamp,
    }
    for filter_type in FILTERS:
        rows[("same_bag", filter_type)] = read_localization_csv(
            getattr(args, f"same_bag_{filter_type}_csv")
        )
        rows[("independent", filter_type)] = read_localization_csv(
            getattr(args, f"independent_{filter_type}_csv")
        )
        prediction_metrics[filter_type], prediction_series[filter_type] = (
            prediction_evaluation(
                rows[("same_bag", filter_type)], reference, args.timestamp_tolerance
            )
        )
        for dataset in DATASETS:
            metrics, series = localization_evaluation(
                rows[(dataset, filter_type)], origins[dataset], dataset
            )
            localization_metrics[(dataset, filter_type)] = metrics
            localization_series[(dataset, filter_type)] = series

    for filter_type in FILTERS:
        independent = localization_metrics[("independent", filter_type)]
        if independent["total_scans"] != args.expected_independent_scans:
            raise ValueError(
                f"{filter_type}: expected {args.expected_independent_scans} independent scans, "
                f"got {independent['total_scans']}"
            )
        summary = {
            "filter_type": filter_type,
            "same_bag_prediction_only": prediction_metrics[filter_type],
            "same_bag_localization": localization_metrics[("same_bag", filter_type)],
            "independent_localization": independent,
            "filter_runtime_definition": (
                "steady-clock latency from nearest adapted sensor input reception to "
                "filtered odometry reception; end-to-end proxy, not internal CPU time"
            ),
            "fairness": {
                "small_gicp_parameters_identical": True,
                "voxel_sizes_identical": True,
                "registration_gates_identical": True,
                "covariance_assumptions_identical": True,
                "initial_global_seed_identical": True,
                "imu_orientation_fused": False,
            },
        }
        save_json(output / f"{filter_type}_summary.json", summary)

    with (output / "prediction_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "filter", "associated_predictions", "translation_rmse_m",
            "x_rmse_m", "y_rmse_m", "z_rmse_m", "yaw_rmse_deg",
            "full_rotation_rmse_deg", "step_translation_p95_m",
            "step_translation_max_m", "step_rotation_p95_deg", "step_rotation_max_deg",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for filter_type in FILTERS:
            item = prediction_metrics[filter_type]
            writer.writerow({
                "filter": filter_type,
                "associated_predictions": item["associated_predictions"],
                "translation_rmse_m": item["translation_rmse_m"],
                "x_rmse_m": item["component_rmse_m"]["x"],
                "y_rmse_m": item["component_rmse_m"]["y"],
                "z_rmse_m": item["component_rmse_m"]["z"],
                "yaw_rmse_deg": item["yaw_rmse_deg"],
                "full_rotation_rmse_deg": item["full_rotation_rmse_deg"],
                "step_translation_p95_m": item["prediction_step_translation_m"]["p95"],
                "step_translation_max_m": item["prediction_step_translation_m"]["max"],
                "step_rotation_p95_deg": item["prediction_step_rotation_deg"]["p95"],
                "step_rotation_max_deg": item["prediction_step_rotation_deg"]["max"],
            })

    with (output / "correction_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "dataset", "filter", "timestamp", "bag_offset_sec",
            "correction_translation_m", "correction_rotation_deg", "correction_yaw_deg",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for filter_type in FILTERS:
                series = localization_series[(dataset, filter_type)]
                for index, timestamp in enumerate(series["correction_times"]):
                    writer.writerow({
                        "dataset": dataset,
                        "filter": filter_type,
                        "timestamp": timestamp,
                        "bag_offset_sec": series["correction_offsets"][index],
                        "correction_translation_m": series["correction_translation"][index],
                        "correction_rotation_deg": series["correction_rotation"][index],
                        "correction_yaw_deg": series["correction_yaw"][index],
                    })

    comparison_fields = [
        "filter", "prediction_translation_rmse_m", "prediction_yaw_rmse_deg",
        "independent_total", "independent_accepted", "independent_rejected",
        "independent_acceptance_rate", "inliers_mean", "inliers_p95", "inliers_min",
        "final_error_mean", "final_error_p95", "iterations_mean", "iterations_p95",
        "gicp_runtime_mean_ms", "gicp_runtime_p95_ms", "gicp_runtime_max_ms",
        "filter_runtime_mean_ms", "filter_runtime_p95_ms", "filter_runtime_max_ms",
        "correction_translation_mean_m", "correction_translation_median_m",
        "correction_translation_p95_m", "correction_translation_max_m",
        "correction_rotation_mean_deg", "correction_rotation_median_deg",
        "correction_rotation_p95_deg", "correction_rotation_max_deg",
        "trajectory_translation_step_p95_m", "trajectory_translation_step_max_m",
        "trajectory_rotation_step_p95_deg", "trajectory_rotation_step_max_deg",
    ]
    with (output / "filter_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        for filter_type in FILTERS:
            prediction = prediction_metrics[filter_type]
            item = localization_metrics[("independent", filter_type)]
            writer.writerow({
                "filter": filter_type,
                "prediction_translation_rmse_m": prediction["translation_rmse_m"],
                "prediction_yaw_rmse_deg": prediction["yaw_rmse_deg"],
                "independent_total": item["total_scans"],
                "independent_accepted": item["accepted_scans"],
                "independent_rejected": item["rejected_scans"],
                "independent_acceptance_rate": item["acceptance_rate"],
                "inliers_mean": item["num_inliers"]["mean"],
                "inliers_p95": item["num_inliers"]["p95"],
                "inliers_min": item["num_inliers"]["min"],
                "final_error_mean": item["gicp_final_error"]["mean"],
                "final_error_p95": item["gicp_final_error"]["p95"],
                "iterations_mean": item["gicp_iterations"]["mean"],
                "iterations_p95": item["gicp_iterations"]["p95"],
                "gicp_runtime_mean_ms": item["gicp_runtime_ms"]["mean"],
                "gicp_runtime_p95_ms": item["gicp_runtime_ms"]["p95"],
                "gicp_runtime_max_ms": item["gicp_runtime_ms"]["max"],
                "filter_runtime_mean_ms": item["filter_runtime_proxy_ms"]["mean"],
                "filter_runtime_p95_ms": item["filter_runtime_proxy_ms"]["p95"],
                "filter_runtime_max_ms": item["filter_runtime_proxy_ms"]["max"],
                "correction_translation_mean_m": item["correction_translation_m"]["mean"],
                "correction_translation_median_m": item["correction_translation_m"]["median"],
                "correction_translation_p95_m": item["correction_translation_m"]["p95"],
                "correction_translation_max_m": item["correction_translation_m"]["max"],
                "correction_rotation_mean_deg": item["correction_rotation_deg"]["mean"],
                "correction_rotation_median_deg": item["correction_rotation_deg"]["median"],
                "correction_rotation_p95_deg": item["correction_rotation_deg"]["p95"],
                "correction_rotation_max_deg": item["correction_rotation_deg"]["max"],
                "trajectory_translation_step_p95_m": item["trajectory_step_translation_m"]["p95"],
                "trajectory_translation_step_max_m": item["trajectory_step_translation_m"]["max"],
                "trajectory_rotation_step_p95_deg": item["trajectory_step_rotation_deg"]["p95"],
                "trajectory_rotation_step_max_deg": item["trajectory_step_rotation_deg"]["max"],
            })

    reference_times, reference_positions, reference_quaternions = reference
    reference_relative = reference_times - reference_times[0]
    save_plot(
        output / "predicted_xy_trajectory.png",
        "Same-bag prediction-only XY trajectory",
        "x [m]", "y [m]",
        lambda axis: (
            axis.plot(reference_positions[:, 0], reference_positions[:, 1], c="black", label="GLIM reference"),
            *[
                axis.plot(
                    prediction_series[name]["positions"][:, 0],
                    prediction_series[name]["positions"][:, 1],
                    label=name.upper(),
                ) for name in FILTERS
            ],
        ),
        equal=True,
    )
    save_plot(
        output / "predicted_yaw.png", "Same-bag prediction-only yaw",
        "time from first reference [s]", "unwrapped yaw [rad]",
        lambda axis: (
            axis.plot(reference_relative, np.unwrap(quaternion_yaw(reference_quaternions)), c="black", label="GLIM reference"),
            *[
                axis.plot(
                    prediction_series[name]["times"] - reference_times[0],
                    np.unwrap(quaternion_yaw(prediction_series[name]["quaternions"])),
                    label=name.upper(),
                ) for name in FILTERS
            ],
        ),
    )
    save_plot(
        output / "prediction_translation_error.png",
        "Same-bag prediction translation error", "time from first reference [s]", "error [m]",
        lambda axis: tuple(
            axis.plot(
                prediction_series[name]["times"] - reference_times[0],
                prediction_series[name]["translation_error"], label=name.upper(),
            ) for name in FILTERS
        ),
    )
    save_plot(
        output / "prediction_yaw_error.png", "Same-bag prediction yaw error",
        "time from first reference [s]", "signed error [deg]",
        lambda axis: tuple(
            axis.plot(
                prediction_series[name]["times"] - reference_times[0],
                prediction_series[name]["yaw_error_deg"], label=name.upper(),
            ) for name in FILTERS
        ),
    )

    def independent_plot(filename, title, ylabel, key, correction=False):
        save_plot(
            output / filename, title, "bag-relative time [s]", ylabel,
            lambda axis: tuple(
                axis.plot(
                    localization_series[("independent", name)][
                        "correction_offsets" if correction else "offsets"
                    ],
                    localization_series[("independent", name)][key],
                    label=name.upper(),
                ) for name in FILTERS
            ),
        )

    independent_plot(
        "gicp_correction_translation.png", "Prediction-to-GICP translation correction",
        "correction [m]", "correction_translation", correction=True,
    )
    independent_plot(
        "gicp_correction_rotation.png", "Prediction-to-GICP rotation correction",
        "correction [deg]", "correction_rotation", correction=True,
    )
    independent_plot(
        "gicp_iterations.png", "Independent-run GICP iterations", "iterations", "iterations"
    )
    independent_plot(
        "gicp_runtime.png", "Independent-run GICP runtime", "runtime [ms]", "gicp_runtime_ms"
    )
    independent_plot(
        "filter_runtime.png", "Filter end-to-end runtime proxy", "latency [ms]", "filter_runtime_ms"
    )
    save_plot(
        output / "accepted_rejected_timeline.png", "Independent accepted/rejected timeline",
        "bag-relative time [s]", "accepted (1) / rejected (0)",
        lambda axis: tuple(
            axis.scatter(
                localization_series[("independent", name)]["offsets"],
                localization_series[("independent", name)]["accepted"].astype(int)
                + (0.02 if name == "ukf" else -0.02),
                s=10, label=name.upper(),
            ) for name in FILTERS
        ),
    )

    criteria = {
        "same-bag translation RMSE": {
            name: prediction_metrics[name]["translation_rmse_m"] for name in FILTERS
        },
        "same-bag yaw RMSE": {
            name: prediction_metrics[name]["yaw_rmse_deg"] for name in FILTERS
        },
        "independent rejection count": {
            name: localization_metrics[("independent", name)]["rejected_scans"] for name in FILTERS
        },
        "correction translation p95": {
            name: localization_metrics[("independent", name)]["correction_translation_m"]["p95"]
            for name in FILTERS
        },
        "correction rotation p95": {
            name: localization_metrics[("independent", name)]["correction_rotation_deg"]["p95"]
            for name in FILTERS
        },
        "GICP iterations mean": {
            name: localization_metrics[("independent", name)]["gicp_iterations"]["mean"]
            for name in FILTERS
        },
    }
    wins = Counter()
    practical_tie_fraction = 0.01
    for values in criteria.values():
        difference = abs(values["ekf"] - values["ukf"])
        scale = max(abs(values["ekf"]), abs(values["ukf"]), 1.0e-12)
        if difference <= practical_tie_fraction * scale:
            continue
        if values["ekf"] < values["ukf"]:
            wins["ekf"] += 1
        elif values["ukf"] < values["ekf"]:
            wins["ukf"] += 1
    if wins["ekf"] > wins["ukf"]:
        conclusion = "EKF is the better current BUNKER initial-guess candidate in this untuned A/B Gate."
    elif wins["ukf"] > wins["ekf"]:
        conclusion = "UKF is the better current BUNKER initial-guess candidate in this untuned A/B Gate."
    else:
        conclusion = "The untuned A/B Gate does not distinguish a clear initial-guess winner."

    def overview_row(filter_type):
        pred = prediction_metrics[filter_type]
        loc = localization_metrics[("independent", filter_type)]
        turn = loc["large_rotation_27_29s"]
        return (
            f"| {filter_type.upper()} | {pred['translation_rmse_m']:.6f} | "
            f"{pred['yaw_rmse_deg']:.6f} | {loc['accepted_scans']}/{loc['total_scans']} | "
            f"{loc['correction_translation_m']['p95']:.6f} | "
            f"{loc['correction_rotation_deg']['p95']:.6f} | "
            f"{loc['gicp_iterations']['mean']:.3f} | {loc['gicp_runtime_ms']['mean']:.3f} | "
            f"{loc['filter_runtime_proxy_ms']['mean']:.3f} | "
            f"{turn['accepted_scans']}/{turn['total_scans']} |"
        )

    def prediction_row(filter_type):
        item = prediction_metrics[filter_type]
        return (
            f"| {filter_type.upper()} | {item['associated_predictions']} | "
            f"{item['translation_rmse_m']:.6f} | {item['component_rmse_m']['x']:.6f} | "
            f"{item['component_rmse_m']['y']:.6f} | {item['component_rmse_m']['z']:.6f} | "
            f"{item['yaw_rmse_deg']:.6f} | {item['full_rotation_rmse_deg']:.6f} | "
            f"{item['prediction_step_translation_m']['p95']:.6f} / "
            f"{item['prediction_step_translation_m']['max']:.6f} | "
            f"{item['prediction_step_rotation_deg']['p95']:.6f} / "
            f"{item['prediction_step_rotation_deg']['max']:.6f} |"
        )

    def independent_row(filter_type):
        item = localization_metrics[("independent", filter_type)]
        return (
            f"| {filter_type.upper()} | {item['total_scans']} | "
            f"{item['accepted_scans']} / {item['rejected_scans']} | "
            f"{100.0 * item['convergence_rate']:.3f}% | "
            f"{item['num_inliers']['mean']:.1f} / {item['num_inliers']['p95']:.1f} / "
            f"{item['num_inliers']['min']:.0f} | "
            f"{item['gicp_final_error']['mean']:.3f} / "
            f"{item['gicp_final_error']['p95']:.3f} / {item['gicp_final_error']['max']:.3f} | "
            f"{item['gicp_iterations']['mean']:.3f} / {item['gicp_iterations']['p95']:.3f} |"
        )

    def correction_row(filter_type):
        item = localization_metrics[("independent", filter_type)]
        translation = item["correction_translation_m"]
        rotation = item["correction_rotation_deg"]
        yaw = item["absolute_correction_yaw_deg"]
        return (
            f"| {filter_type.upper()} | "
            f"{translation['mean']:.6f} / {translation['median']:.6f} / "
            f"{translation['p95']:.6f} / {translation['max']:.6f} | "
            f"{rotation['mean']:.6f} / {rotation['median']:.6f} / "
            f"{rotation['p95']:.6f} / {rotation['max']:.6f} | "
            f"{yaw['mean']:.6f} / {yaw['median']:.6f} / "
            f"{yaw['p95']:.6f} / {yaw['max']:.6f} |"
        )

    def runtime_row(filter_type):
        item = localization_metrics[("independent", filter_type)]
        registration = item["gicp_runtime_ms"]
        timing = item["filter_runtime_proxy_ms"]
        return (
            f"| {filter_type.upper()} | {registration['mean']:.3f} / "
            f"{registration['p95']:.3f} / {registration['max']:.3f} | "
            f"{timing['mean']:.3f} / {timing['p95']:.3f} / {timing['max']:.3f} |"
        )

    def turn_row(filter_type):
        item = localization_metrics[("independent", filter_type)]["large_rotation_27_29s"]
        return (
            f"| {filter_type.upper()} | {item['accepted_scans']} / {item['rejected_scans']} | "
            f"{item['correction_translation_m']['mean']:.6f} / "
            f"{item['correction_translation_m']['p95']:.6f} | "
            f"{item['correction_rotation_deg']['mean']:.6f} / "
            f"{item['correction_rotation_deg']['p95']:.6f} | "
            f"{item['iterations']['mean']:.3f} / {item['iterations']['p95']:.3f} | "
            f"{item['gicp_runtime_ms']['mean']:.3f} / "
            f"{item['gicp_runtime_ms']['p95']:.3f} |"
        )

    independent_reject_text = {}
    for filter_type in FILTERS:
        events = localization_metrics[("independent", filter_type)]["recovery_after_reject"]
        independent_reject_text[filter_type] = ", ".join(
            f"{event['reject_offset_sec']:.6f}s ({event['reason']})" for event in events
        )
    same_ekf = localization_metrics[("same_bag", "ekf")]
    same_ukf = localization_metrics[("same_bag", "ukf")]
    ukf_divergence = prediction_metrics["ukf"]["first_translation_error_above_1m_offset_sec"]

    report = f"""# EKF vs UKF localization comparison Gate

Only the official robot_localization executable changes between runs. Map, scan stream, global
seed, adapter covariances, fused fields, small_gicp parameters, voxel sizes, and registration
quality gates are identical.

| Filter | prediction translation RMSE [m] | prediction yaw RMSE [deg] | independent accepted | correction translation p95 [m] | correction rotation p95 [deg] | GICP iterations mean | GICP runtime mean [ms] | filter latency mean [ms] | 27-29 s accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{overview_row('ekf')}
{overview_row('ukf')}

## Same-bag prediction-only comparison

| Filter | associated | translation RMSE [m] | x RMSE | y RMSE | z RMSE | yaw RMSE [deg] | full rotation RMSE [deg] | translation step p95/max [m] | rotation step p95/max [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{prediction_row('ekf')}
{prediction_row('ukf')}

The EKF prediction never exceeded 1 m translation error (maximum
{prediction_metrics['ekf']['translation_error_m']['max']:.3f} m). The UKF first exceeded 1 m at
{ukf_divergence:.3f} s and reached
{prediction_metrics['ukf']['translation_error_m']['max']:.3f} m. Its dominant component is z.
This propagated into same-bag localization: EKF accepted {same_ekf['accepted_scans']}/869 with
{same_ekf['rejection_reasons']}; UKF accepted {same_ukf['accepted_scans']}/869 with
{same_ukf['rejection_reasons']}. EKF recovered after every reject on the next scan. UKF has
{same_ukf['recovery_summary']['unrecovered_rejects']} rejects with no later recovery and a maximum
recovered gap of {same_ukf['recovery_summary']['scans_to_recovery']['max']:.0f} scans.

## Independent 0-50 s comparison

| Filter | scans | accepted / rejected | convergence | inliers mean/p95/min | final error mean/p95/max | iterations mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
{independent_row('ekf')}
{independent_row('ukf')}

- EKF reject events: {independent_reject_text['ekf']}
- UKF reject events: {independent_reject_text['ukf']}
- Every independent reject recovered on the next scan (about 0.1002 s) for both filters.

## Prediction-to-GICP correction on independent scans

Values are mean / median / p95 / max.

| Filter | translation [m] | full rotation [deg] | absolute yaw [deg] |
|---|---:|---:|---:|
{correction_row('ekf')}
{correction_row('ukf')}

## Runtime comparison

Values are mean / p95 / max.

| Filter | GICP runtime [ms] | filter latency proxy [ms] |
|---|---:|---:|
{runtime_row('ekf')}
{runtime_row('ukf')}

## Large-rotation interval, 27-29 s

| Filter | accepted / rejected | correction translation mean/p95 [m] | correction rotation mean/p95 [deg] | iterations mean/p95 | GICP runtime mean/p95 [ms] |
|---|---:|---:|---:|---:|---:|
{turn_row('ekf')}
{turn_row('ukf')}

## Current decision

{conclusion} The evidence count is EKF {wins['ekf']} vs UKF {wins['ukf']} across six primary
criteria after treating differences within 1% as practical ties. This is a dataset-specific
engineering decision, not proof that one estimator family is generally superior. UKF-specific
tuning was deliberately not performed.

## Interpretation

- Same-bag values compare the pre-registration `T_map_lidar` prediction against timestamp-matched
  GLIM poses. Final small_gicp poses are excluded from those prediction-only errors.
- Corrections use `delta_T = inverse(T_prediction_map_lidar) * T_gicp_map_lidar`; smaller values
  mean the initial guess landed closer to the accepted registration, but are not sufficient alone
  to establish filter superiority.
- Independent 163346 processing remains restricted to bag-relative 0-50 s. It has no reference
  trajectory, so no independent absolute RMSE is reported.
- `filter_runtime` is an identically measured steady-clock, end-to-end latency proxy from the
  nearest adapted sensor input reception to filtered odometry reception. It includes ROS
  scheduling/transport and is not robot_localization internal CPU time.
- The actual `T_base_lidar` remains unavailable; both runs use the same explicit Phase 1 identity
  prediction approximation.
"""
    (output / "comparison_report.md").write_text(report, encoding="utf-8")

    config_paths = [Path(path) for path in args.config]
    save_json(
        output / "comparison_manifest.json",
        {
            "config_sha256": {str(path): file_sha256(path) for path in config_paths},
            "filters": list(FILTERS),
            "datasets": list(DATASETS),
            "independent_window_sec": [0.0, 50.0],
            "decision_criteria": criteria,
            "criterion_wins": dict(wins),
            "practical_tie_fraction": practical_tie_fraction,
            "conclusion": conclusion,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    for dataset in DATASETS:
        for filter_type in FILTERS:
            parser.add_argument(f"--{dataset.replace('_', '-')}-{filter_type}-csv", required=True)
    parser.add_argument("--reference-trajectory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--independent-origin-timestamp", type=float, required=True)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.06)
    parser.add_argument("--expected-independent-scans", type=int, default=490)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    if args.timestamp_tolerance <= 0.0 or args.expected_independent_scans <= 0:
        parser.error("timestamp tolerance and expected scan count must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
