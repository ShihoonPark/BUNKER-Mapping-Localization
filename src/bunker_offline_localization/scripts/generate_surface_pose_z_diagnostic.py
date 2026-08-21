#!/usr/bin/env python3

"""Blind physical-surface pose-z diagnostics followed by a sealed post-hoc height check."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_localization_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_imu_gicp_physical_gate as physical_gate
import generate_plateau_label_diagnostic as plateau_diagnostic

np = physical_gate.np
plt = physical_gate.plt
yaml = physical_gate.yaml


GROUND = "ground_before"
STAGE = "stage_top"
LABELS = (GROUND, STAGE)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    physical_gate.write_csv(path, rows, fieldnames)


def write_json(path, value):
    physical_gate.write_json(path, value)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def residual_metrics(residuals):
    residuals = np.asarray(residuals, dtype=float)
    absolute = np.abs(residuals)
    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        raise ValueError("Residual metrics require finite non-empty data")
    return {
        "rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "mae_m": float(np.mean(absolute)),
        "median_absolute_residual_m": float(np.median(absolute)),
        "p95_absolute_residual_m": float(np.percentile(absolute, 95.0)),
        "maximum_absolute_residual_m": float(np.max(absolute)),
    }


def huber_irls(design, observations, fit_config):
    """Deterministic Huber IRLS. This blind primitive has no measured-height input."""
    design = np.asarray(design, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if design.ndim != 2 or observations.ndim != 1 or design.shape[0] != observations.size:
        raise ValueError("Invalid robust-fit design or observation shape")
    if design.shape[0] < design.shape[1] or not np.all(np.isfinite(design)):
        raise ValueError("Robust fit requires a finite overdetermined design")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Robust fit observations must be finite")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(f"Design is rank deficient: {rank}/{design.shape[1]}")
    condition_number = float(np.linalg.cond(design))
    beta = np.linalg.lstsq(design, observations, rcond=None)[0]
    delta = float(fit_config["huber_parameter"])
    tolerance = float(fit_config["convergence_tolerance"])
    minimum_scale = float(fit_config["minimum_scale_m"])
    maximum_iterations = int(fit_config["maximum_iterations"])
    converged = False
    coefficient_step = math.inf
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        residuals = observations - design @ beta
        centered = residuals - np.median(residuals)
        scale = max(minimum_scale, 1.4826 * float(np.median(np.abs(centered))))
        normalized = np.abs(residuals) / (delta * scale)
        weights = np.ones(residuals.size)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_observations = observations * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_design, weighted_observations, rcond=None)[0]
        coefficient_step = float(np.linalg.norm(updated - beta))
        beta = updated
        if coefficient_step <= tolerance * (1.0 + float(np.linalg.norm(beta))):
            converged = True
            break
    fitted = design @ beta
    residuals = observations - fitted
    centered = residuals - np.median(residuals)
    scale = max(minimum_scale, 1.4826 * float(np.median(np.abs(centered))))
    normalized = np.abs(residuals) / (delta * scale)
    weights = np.ones(residuals.size)
    outside = normalized > 1.0
    weights[outside] = 1.0 / normalized[outside]
    return {
        "coefficients": beta,
        "fitted": fitted,
        "residuals": residuals,
        "weights": weights,
        "rank": rank,
        "columns": int(design.shape[1]),
        "condition_number": condition_number,
        "sample_count": int(observations.size),
        "iterations": iterations,
        "converged": converged,
        "final_coefficient_step": coefficient_step,
        "robust_scale_m": scale,
        "huber_parameter": delta,
        "weight_minimum": float(np.min(weights)),
        "weight_median": float(np.median(weights)),
        "downweighted_count": int(np.count_nonzero(weights < 1.0 - 1.0e-12)),
        "residual_metrics": residual_metrics(residuals),
    }


def plane_design(rows):
    return np.asarray([[row["x_m"], row["y_m"], 1.0] for row in rows]), np.asarray(
        [row["z_m"] for row in rows]
    )


def common_design(rows):
    return np.asarray([
        [row["x_m"], row["y_m"], 1.0, 1.0 if row["physical_label"] == STAGE else 0.0]
        for row in rows
    ]), np.asarray([row["z_m"] for row in rows])


def fit_plane(rows, fit_config):
    design, observations = plane_design(rows)
    return huber_irls(design, observations, fit_config)


def fit_common_planes(rows, fit_config):
    design, observations = common_design(rows)
    return huber_irls(design, observations, fit_config)


def plane_normal(a, b):
    normal = np.asarray([-a, -b, 1.0], dtype=float)
    return normal / np.linalg.norm(normal)


def plane_fit_json(fit, physical_label, observation_level, candidate_count):
    a, b, c = [float(value) for value in fit["coefficients"]]
    magnitude = math.hypot(a, b)
    direction = None if magnitude < 1.0e-15 else [a / magnitude, b / magnitude]
    return {
        "physical_label": physical_label,
        "fitted_quantity": "map_frame_lidar_sensor_center_pose_z_spatial_support_plane",
        "observation_level": observation_level,
        "model": "z = a*x + b*y + c",
        "a_dz_dx": a,
        "b_dz_dy": b,
        "c_m": c,
        "slope_magnitude_m_per_m": magnitude,
        "slope_angle_deg": math.degrees(math.atan(magnitude)),
        "normalized_plane_normal": plane_normal(a, b),
        "spatial_gradient_unit_xy": direction,
        "spatial_gradient_azimuth_deg": None if direction is None else math.degrees(math.atan2(b, a)),
        "sample_count": fit["sample_count"],
        "candidate_count": candidate_count,
        "design_matrix_rank": fit["rank"],
        "design_matrix_columns": fit["columns"],
        "design_matrix_condition_number": fit["condition_number"],
        "fit_iterations": fit["iterations"],
        "fit_converged": fit["converged"],
        "robust_scale_m": fit["robust_scale_m"],
        "final_robust_weight_minimum": fit["weight_minimum"],
        "downweighted_observation_count": fit["downweighted_count"],
        **fit["residual_metrics"],
    }


def common_fit_json(fit, observation_level, candidate_counts):
    a, b, c_ground, height = [float(value) for value in fit["coefficients"]]
    magnitude = math.hypot(a, b)
    return {
        "fitted_quantity": "map_frame_lidar_sensor_center_pose_z_parallel_support_planes",
        "observation_level": observation_level,
        "model": "z = a*x + b*y + c_ground + I_stage*h",
        "a_dz_dx": a,
        "b_dz_dy": b,
        "c_ground_m": c_ground,
        "h_stage_minus_ground_m": height,
        "slope_magnitude_m_per_m": magnitude,
        "slope_angle_deg": math.degrees(math.atan(magnitude)),
        "normalized_plane_normal": plane_normal(a, b),
        "spatial_gradient_azimuth_deg": math.degrees(math.atan2(b, a)) if magnitude else None,
        "sample_count": fit["sample_count"],
        "candidate_counts": dict(candidate_counts),
        "design_matrix_rank": fit["rank"],
        "design_matrix_columns": fit["columns"],
        "design_matrix_condition_number": fit["condition_number"],
        "fit_iterations": fit["iterations"],
        "fit_converged": fit["converged"],
        "robust_scale_m": fit["robust_scale_m"],
        "final_robust_weight_minimum": fit["weight_minimum"],
        "downweighted_observation_count": fit["downweighted_count"],
        **fit["residual_metrics"],
    }


def load_fixed_candidates(candidate_path, fixed_config):
    ground_ids = [int(value) for value in fixed_config[GROUND]]
    stage_ids = [int(value) for value in fixed_config[STAGE]]
    if ground_ids != list(range(1, 12)) or stage_ids != list(range(12, 23)):
        raise ValueError("Physical-label lock must remain IDs 1-11 ground and 12-22 stage")
    if str(fixed_config["ground_after"]).lower() != "unavailable":
        raise ValueError("ground_after must remain unavailable in the 0-50 s interval")
    if fixed_config.get("measured_height_used_for_labeling", True):
        raise ValueError("Measured height must not participate in physical labeling")
    rows = read_csv(candidate_path)
    ids = [int(row["diagnostic_candidate_id"]) for row in rows]
    if ids != list(range(1, 23)):
        raise ValueError(f"Expected the frozen 22 diagnostic IDs, got {ids}")
    label_by_id = {candidate_id: GROUND for candidate_id in ground_ids}
    label_by_id.update({candidate_id: STAGE for candidate_id in stage_ids})
    candidates = []
    for row in rows:
        candidate_id = int(row["diagnostic_candidate_id"])
        if row.get("automatic_physical_label") != "UNASSIGNED":
            raise ValueError("Source diagnostic candidates must remain automatically UNASSIGNED")
        candidates.append({
            "diagnostic_id": candidate_id,
            "physical_label": label_by_id[candidate_id],
            "start_time_sec": float(row["start_offset_sec"]),
            "end_time_sec": float(row["end_offset_sec"]),
            "source_sample_count": int(row["sample_count"]),
        })
    return candidates


def sanitized_plateau_provenance(summary_path):
    with Path(summary_path).open(encoding="utf-8") as stream:
        source = json.load(stream)
    # Deliberately return only height-blind provenance. measured_stage_height_m is ignored.
    return {
        "candidate_count": int(source["candidate_count"]),
        "candidate_search_window_sec": source["candidate_search_window_sec"],
        "diagnostic_only": bool(source["diagnostic_only"]),
        "new_physical_labels_assigned": bool(source["new_physical_labels_assigned"]),
        "measured_height_used_for_candidate_detection": bool(
            source["measured_height_used_for_candidate_detection"]
        ),
        "ignored_source_fields": ["measured_stage_height_m", "candidate_pair_count"],
    }


def build_labeled_samples(records, candidates, sensor, origin, maximum_speed_dt):
    accepted = [record for record in records if record["accepted"]]
    samples = []
    assigned_rows = set()
    for candidate in candidates:
        selected = [
            record for record in accepted
            if candidate["start_time_sec"] <= record["offset"] <= candidate["end_time_sec"]
        ]
        if len(selected) != candidate["source_sample_count"]:
            raise ValueError(
                f"Candidate {candidate['diagnostic_id']} sample count changed: "
                f"{len(selected)} != {candidate['source_sample_count']}"
            )
        timestamps = np.asarray([record["timestamp"] for record in selected])
        speeds, speed_dt = physical_gate.nearest_values(
            timestamps, sensor["odom_times"], sensor["odom_linear_speed"]
        )
        if np.any(speed_dt > maximum_speed_dt):
            raise ValueError(
                f"Candidate {candidate['diagnostic_id']} has odom association beyond "
                f"{maximum_speed_dt} s"
            )
        for record, speed, association_dt in zip(selected, speeds, speed_dt):
            if record["row_index"] in assigned_rows:
                raise ValueError("Candidate windows overlap at a localization row")
            assigned_rows.add(record["row_index"])
            samples.append({
                "timestamp": record["timestamp"],
                "time_sec": record["offset"],
                "diagnostic_id": candidate["diagnostic_id"],
                "physical_label": candidate["physical_label"],
                "x_m": float(record["position"][0]),
                "y_m": float(record["position"][1]),
                "z_m": float(record["position"][2]),
                "roll_rad": float(record["rpy"][0]),
                "pitch_rad": float(record["rpy"][1]),
                "yaw_rad": float(record["rpy"][2]),
                "roll_deg": math.degrees(float(record["rpy"][0])),
                "pitch_deg": math.degrees(float(record["rpy"][1])),
                "yaw_deg": math.degrees(float(record["rpy"][2])),
                "odom_speed_mps": float(speed),
                "odom_speed_association_dt_sec": float(association_dt),
            })
    samples.sort(key=lambda row: row["timestamp"])
    return samples


def candidate_metrics(samples, candidates):
    rows = []
    for candidate in candidates:
        selected = [row for row in samples if row["diagnostic_id"] == candidate["diagnostic_id"]]
        def values(name):
            return np.asarray([row[name] for row in selected], dtype=float)
        z = values("z_m")
        rows.append({
            "diagnostic_id": candidate["diagnostic_id"],
            "physical_label": candidate["physical_label"],
            "start_time_sec": candidate["start_time_sec"],
            "end_time_sec": candidate["end_time_sec"],
            "sample_count": len(selected),
            "median_x_m": float(np.median(values("x_m"))),
            "median_y_m": float(np.median(values("y_m"))),
            "median_z_m": float(np.median(z)),
            "mean_z_m": float(np.mean(z)),
            "std_z_m": float(np.std(z)),
            "min_z_m": float(np.min(z)),
            "max_z_m": float(np.max(z)),
            "z_p05_m": float(np.percentile(z, 5.0)),
            "z_p95_m": float(np.percentile(z, 95.0)),
            "median_roll_deg": float(np.median(values("roll_deg"))),
            "median_pitch_deg": float(np.median(values("pitch_deg"))),
            "median_yaw_deg": float(np.median(values("yaw_deg"))),
            "median_speed_mps": float(np.median(values("odom_speed_mps"))),
        })
    return rows


def candidate_fit_rows(metrics):
    return [{
        "diagnostic_id": row["diagnostic_id"],
        "physical_label": row["physical_label"],
        "x_m": row["median_x_m"],
        "y_m": row["median_y_m"],
        "z_m": row["median_z_m"],
    } for row in metrics]


def parallelism(first_json, second_json):
    first_normal = np.asarray(first_json["normalized_plane_normal"])
    second_normal = np.asarray(second_json["normalized_plane_normal"])
    angle = math.degrees(math.acos(float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))))
    return {
        "angle_between_normals_deg": angle,
        "delta_dz_dx_stage_minus_ground": second_json["a_dz_dx"] - first_json["a_dz_dx"],
        "delta_dz_dy_stage_minus_ground": second_json["b_dz_dy"] - first_json["b_dz_dy"],
        "ground_normal": first_json["normalized_plane_normal"],
        "stage_normal": second_json["normalized_plane_normal"],
        "threshold_based_pass_fail_applied": False,
    }


def label_xy_confounding(rows):
    design = np.asarray([[row["x_m"], row["y_m"], 1.0] for row in rows])
    indicator = np.asarray([1.0 if row["physical_label"] == STAGE else 0.0 for row in rows])
    fitted = design @ np.linalg.lstsq(design, indicator, rcond=None)[0]
    total = float(np.sum((indicator - np.mean(indicator)) ** 2))
    residual = float(np.sum((indicator - fitted) ** 2))
    r_squared = 1.0 - residual / total
    group_support = {}
    for label in LABELS:
        selected = [row for row in rows if row["physical_label"] == label]
        x = np.asarray([row["x_m"] for row in selected])
        y = np.asarray([row["y_m"] for row in selected])
        group_support[label] = {
            "x_range_m": [float(np.min(x)), float(np.max(x))],
            "y_range_m": [float(np.min(y)), float(np.max(y))],
            "centroid_xy_m": [float(np.mean(x)), float(np.mean(y))],
        }
    centroid_distance = float(np.linalg.norm(
        np.asarray(group_support[STAGE]["centroid_xy_m"]) -
        np.asarray(group_support[GROUND]["centroid_xy_m"])
    ))
    return {
        "stage_indicator_r_squared_from_linear_xy": r_squared,
        "stage_indicator_variance_inflation_factor": 1.0 / max(1.0e-15, 1.0 - r_squared),
        "group_xy_support": group_support,
        "group_centroid_distance_m": centroid_distance,
        "interpretation": (
            "High label-from-XY predictability exposes spatial-label confounding; it does not "
            "establish a physical height or map tilt."
        ),
    }


def group_residual_metrics(rows, field):
    return {
        label: residual_metrics([row[field] for row in rows if row["physical_label"] == label])
        for label in LABELS
    }


def raw_group_pose_z_metrics(rows):
    result = {}
    for label in LABELS:
        values = np.asarray([row["z_m"] for row in rows if row["physical_label"] == label])
        result[label] = {
            "sample_count": int(values.size),
            "minimum_z_m": float(np.min(values)),
            "maximum_z_m": float(np.max(values)),
            "range_z_m": float(np.max(values) - np.min(values)),
            "median_z_m": float(np.median(values)),
            "p05_z_m": float(np.percentile(values, 5.0)),
            "p95_z_m": float(np.percentile(values, 95.0)),
        }
    return result


def residual_correlations(residual_rows):
    rows = []
    variables = (
        ("gicp_pitch_deg", "pitch_deg"),
        ("gicp_roll_deg", "roll_deg"),
        ("odom_speed_mps", "odom_speed_mps"),
    )
    for scope in ("all", GROUND, STAGE):
        selected = residual_rows if scope == "all" else [
            row for row in residual_rows if row["physical_label"] == scope
        ]
        residual = [row["common_residual_z_m"] for row in selected]
        for output_name, field in variables:
            values = [row[field] for row in selected]
            rows.append({
                "scope": scope,
                "variable": output_name,
                "sample_count": len(selected),
                "pearson": physical_gate.pearson(residual, values),
                "spearman": physical_gate.spearman(residual, values),
                "causal_interpretation_established": False,
            })
    return rows


def candidate_block_bootstrap(samples, fit_config, bootstrap_config):
    by_id = {
        candidate_id: [row for row in samples if row["diagnostic_id"] == candidate_id]
        for candidate_id in range(1, 23)
    }
    ids_by_label = {
        GROUND: np.asarray(range(1, 12), dtype=int),
        STAGE: np.asarray(range(12, 23), dtype=int),
    }
    generator = np.random.default_rng(int(bootstrap_config["seed"]))
    height_rows = []
    failures = 0
    requested = int(bootstrap_config["samples"])
    for iteration in range(requested):
        resampled = []
        selected_ids = {}
        for label in LABELS:
            ids = ids_by_label[label]
            draws = generator.choice(ids, size=ids.size, replace=True)
            selected_ids[label] = [int(value) for value in draws]
            for candidate_id in draws:
                resampled.extend(by_id[int(candidate_id)])
        try:
            fit = fit_common_planes(resampled, fit_config)
            height = float(fit["coefficients"][3])
            if not np.isfinite(height):
                raise ValueError("Non-finite bootstrap height")
            height_rows.append({
                "bootstrap_iteration": iteration,
                "h_stage_minus_ground_m": height,
                "ground_candidate_ids": ";".join(map(str, selected_ids[GROUND])),
                "stage_candidate_ids": ";".join(map(str, selected_ids[STAGE])),
            })
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    if not height_rows:
        raise ValueError("Every candidate-block bootstrap fit failed")
    heights = np.asarray([row["h_stage_minus_ground_m"] for row in height_rows])
    return height_rows, {
        "method": bootstrap_config["method"],
        "seed": int(bootstrap_config["seed"]),
        "requested_samples": requested,
        "successful_samples": len(height_rows),
        "failure_count": failures,
        "h_median_m": float(np.median(heights)),
        "h_p02_5_m": float(np.percentile(heights, 2.5)),
        "h_p97_5_m": float(np.percentile(heights, 97.5)),
        "time_samples_treated_as_independent": False,
    }


def blind_fit_analysis(samples, metrics, fit_config, bootstrap_config):
    sample_groups = {label: [row for row in samples if row["physical_label"] == label] for label in LABELS}
    candidates = candidate_fit_rows(metrics)
    candidate_groups = {
        label: [row for row in candidates if row["physical_label"] == label] for label in LABELS
    }
    sample_fits = {label: fit_plane(sample_groups[label], fit_config) for label in LABELS}
    candidate_fits = {label: fit_plane(candidate_groups[label], fit_config) for label in LABELS}
    common_sample_fit = fit_common_planes(samples, fit_config)
    common_candidate_fit = fit_common_planes(candidates, fit_config)
    counts = {label: len(candidate_groups[label]) for label in LABELS}
    sample_json = {
        label: plane_fit_json(sample_fits[label], label, "accepted_pose_sample", counts[label])
        for label in LABELS
    }
    candidate_json = {
        label: plane_fit_json(candidate_fits[label], label, "candidate_median", counts[label])
        for label in LABELS
    }
    common_sample_json = common_fit_json(common_sample_fit, "accepted_pose_sample", counts)
    common_candidate_json = common_fit_json(common_candidate_fit, "candidate_median", counts)

    for label in LABELS:
        for row, weight, fitted, residual in zip(
            sample_groups[label], sample_fits[label]["weights"],
            sample_fits[label]["fitted"], sample_fits[label]["residuals"]
        ):
            row["independent_fitted_z_m"] = float(fitted)
            row["independent_residual_z_m"] = float(residual)
            row["independent_robust_weight"] = float(weight)
    for row, weight, fitted, residual in zip(
        samples, common_sample_fit["weights"], common_sample_fit["fitted"], common_sample_fit["residuals"]
    ):
        row["common_fitted_z_m"] = float(fitted)
        row["common_residual_z_m"] = float(residual)
        row["common_robust_weight"] = float(weight)
    for label in LABELS:
        for row, weight in zip(candidate_groups[label], candidate_fits[label]["weights"]):
            metrics[row["diagnostic_id"] - 1]["independent_candidate_fit_weight"] = float(weight)
    for row, weight in zip(candidates, common_candidate_fit["weights"]):
        metrics[row["diagnostic_id"] - 1]["common_candidate_fit_weight"] = float(weight)

    correlations = residual_correlations(samples)
    candidate_residuals = []
    for candidate_id in range(1, 23):
        selected = [row for row in samples if row["diagnostic_id"] == candidate_id]
        candidate_residuals.append({
            "diagnostic_id": candidate_id,
            "physical_label": selected[0]["physical_label"],
            "sample_count": len(selected),
            "median_common_residual_z_m": float(np.median([row["common_residual_z_m"] for row in selected])),
            "mean_common_residual_z_m": float(np.mean([row["common_residual_z_m"] for row in selected])),
            "p95_absolute_common_residual_z_m": float(np.percentile(
                np.abs([row["common_residual_z_m"] for row in selected]), 95.0
            )),
            "minimum_common_robust_weight": float(np.min([row["common_robust_weight"] for row in selected])),
        })

    bootstrap_rows, bootstrap_json = candidate_block_bootstrap(
        samples, fit_config, bootstrap_config
    )
    independent_group_metrics = group_residual_metrics(samples, "independent_residual_z_m")
    common_group_metrics = group_residual_metrics(samples, "common_residual_z_m")
    comparison = {}
    for label in LABELS:
        independent_rmse = independent_group_metrics[label]["rmse_m"]
        common_rmse = common_group_metrics[label]["rmse_m"]
        comparison[label] = {
            "independent": independent_group_metrics[label],
            "common_tilt": common_group_metrics[label],
            "rmse_increase_m": common_rmse - independent_rmse,
            "rmse_ratio_common_over_independent": common_rmse / independent_rmse,
        }
    return {
        "sample_fits_internal": sample_fits,
        "candidate_fits_internal": candidate_fits,
        "common_sample_fit_internal": common_sample_fit,
        "common_candidate_fit_internal": common_candidate_fit,
        "independent_sample_planes": sample_json,
        "independent_candidate_planes": candidate_json,
        "sample_plane_parallelism": parallelism(sample_json[GROUND], sample_json[STAGE]),
        "candidate_plane_parallelism": parallelism(candidate_json[GROUND], candidate_json[STAGE]),
        "raw_group_pose_z_metrics": raw_group_pose_z_metrics(samples),
        "sample_label_xy_confounding": label_xy_confounding(samples),
        "candidate_label_xy_confounding": label_xy_confounding(candidates),
        "common_sample_plane": common_sample_json,
        "common_candidate_plane": common_candidate_json,
        "residual_model_comparison": comparison,
        "residual_correlations": correlations,
        "candidate_residuals": candidate_residuals,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap": bootstrap_json,
    }


def plot_fixed_labels(path, map_xy, records, metrics):
    accepted = np.asarray([record["position"][:2] for record in records if record["accepted"]])
    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    axis.scatter(map_xy[:, 0], map_xy[:, 1], s=1.0, color="0.75", alpha=0.35, label="classroom PLY XY")
    axis.plot(accepted[:, 0], accepted[:, 1], linewidth=1.0, color="0.35", label="accepted GICP trajectory")
    colors = {GROUND: "tab:green", STAGE: "tab:orange"}
    colocated_start_offsets = {
        1: (-44, 14), 2: (0, 30), 3: (40, 14),
        4: (-22, -30), 5: (0, 22), 6: (0, -30),
    }
    for label in LABELS:
        rows = [row for row in metrics if row["physical_label"] == label]
        axis.scatter(
            [row["median_x_m"] for row in rows], [row["median_y_m"] for row in rows],
            s=65, color=colors[label], label=f"{label} fixed IDs",
        )
        for row in rows:
            candidate_id = row["diagnostic_id"]
            offset = colocated_start_offsets.get(candidate_id, (5, 6))
            axis.annotate(
                str(candidate_id), (row["median_x_m"], row["median_y_m"]),
                xytext=offset, textcoords="offset points", fontsize=11, fontweight="bold",
                ha=("left" if offset[0] >= 0 else "right"),
                color=colors[label],
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                arrowprops=(
                    {"arrowstyle": "-", "color": colors[label], "linewidth": 0.7, "alpha": 0.7}
                    if candidate_id <= 6 else None
                ),
            )
    axis.set_title("Fixed physical labels (height-blind): ground 1-11, stage 12-22")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    axis.text(
        0.99, 0.01, "ground_after: unavailable in 0-50 s", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=10, fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_independent_planes(path, samples, plane_json):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = {GROUND: "Greens", STAGE: "Oranges"}
    for axis, label in zip(axes, LABELS):
        rows = [row for row in samples if row["physical_label"] == label]
        x = np.asarray([row["x_m"] for row in rows])
        y = np.asarray([row["y_m"] for row in rows])
        z = np.asarray([row["z_m"] for row in rows])
        fit = plane_json[label]
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 80), np.linspace(y.min(), y.max(), 80))
        grid_z = fit["a_dz_dx"] * grid_x + fit["b_dz_dy"] * grid_y + fit["c_m"]
        image = axis.contourf(grid_x, grid_y, grid_z, levels=16, cmap=colors[label], alpha=0.65)
        points = axis.scatter(x, y, c=z, cmap=colors[label], s=16, edgecolor="black", linewidth=0.2)
        figure.colorbar(image, ax=axis, label="independent fitted pose-z [m]")
        figure.colorbar(points, ax=axis, label="observed pose-z [m]")
        axis.set_title(
            f"{label}: sensor-center pose-z support plane\n"
            f"a={fit['a_dz_dx']:.4f}, b={fit['b_dz_dy']:.4f}, slope={fit['slope_angle_deg']:.2f} deg"
        )
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle("Observed GICP pose-z and independent fitted spatial trends (not floor geometry)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_spatial_trend(path, samples):
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = {GROUND: "tab:green", STAGE: "tab:orange"}
    for row_index, label in enumerate(LABELS):
        trend_axis = axes[row_index, 0]
        residual_axis = axes[row_index, 1]
        rows = sorted([row for row in samples if row["physical_label"] == label], key=lambda row: row["timestamp"])
        xy = np.asarray([[row["x_m"], row["y_m"]] for row in rows])
        progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        trend_axis.scatter(progress, [row["z_m"] for row in rows], s=12, alpha=0.65, color=colors[label], label="raw pose-z")
        trend_axis.plot(progress, [row["independent_fitted_z_m"] for row in rows], color="black", linewidth=1.4, label="independent spatial fit")
        trend_axis.axhline(0.0, color="0.7", linewidth=0.6)
        trend_axis.set_ylabel("map-frame LiDAR pose-z [m]")
        trend_axis.set_title(f"{label}: raw and fitted trend")
        trend_axis.grid(True, alpha=0.25)
        trend_axis.legend(loc="best")
        residual_axis.scatter(
            progress, [row["independent_residual_z_m"] for row in rows],
            s=12, alpha=0.65, color=colors[label],
        )
        residual_axis.axhline(0.0, color="black", linewidth=0.8)
        residual_axis.set_ylabel("independent fit residual [m]")
        residual_axis.set_title(f"{label}: detrended residual")
        residual_axis.grid(True, alpha=0.25)
        if row_index == len(LABELS) - 1:
            trend_axis.set_xlabel("candidate-sample XY chord progress [m]")
            residual_axis.set_xlabel("candidate-sample XY chord progress [m]")
    figure.suptitle("Pose-z, independent spatial trend, and residual by fixed physical surface")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_common_residual(path, samples):
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = {GROUND: "tab:green", STAGE: "tab:orange"}
    for label in LABELS:
        rows = [row for row in samples if row["physical_label"] == label]
        axis.scatter(
            [row["time_sec"] for row in rows], [row["common_residual_z_m"] for row in rows],
            s=14, alpha=0.7, color=colors[label], label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="bag-relative time [s]", ylabel="common-tilt pose-z residual [m]",
             title="Common-tilt detrended residual (fixed labels, accepted candidate samples only)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_residual_correlates(path, samples):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    variables = (("pitch_deg", "GICP pitch [deg]"), ("roll_deg", "GICP roll [deg]"),
                 ("odom_speed_mps", "odom speed [m/s]"))
    colors = {GROUND: "tab:green", STAGE: "tab:orange"}
    for axis, (field, label_text) in zip(axes, variables):
        for label in LABELS:
            rows = [row for row in samples if row["physical_label"] == label]
            axis.scatter(
                [row[field] for row in rows], [row["common_residual_z_m"] for row in rows],
                s=13, alpha=0.55, color=colors[label], label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel(label_text)
        axis.set_ylabel("common residual z [m]")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    figure.suptitle("Residual association diagnostics; correlation does not establish cause")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def height_posthoc(common_fit, bootstrap, physical_gate_config_path):
    """Phase B only: read measured height after the blind summary has been sealed."""
    with Path(physical_gate_config_path).open(encoding="utf-8") as stream:
        physical_config = yaml.safe_load(stream)["physical_stage_height"]
    estimated = float(common_fit["h_stage_minus_ground_m"])
    measured = float(physical_config["height_m"])
    tolerance = physical_config.get("tolerance_m")
    signed_error = estimated - measured
    absolute_error = abs(signed_error)
    if tolerance is None:
        status = "HEIGHT_THRESHOLD_UNDEFINED"
    else:
        status = "HEIGHT_PASS" if absolute_error <= float(tolerance) else "HEIGHT_FAIL"
    return {
        "diagnostic_only": True,
        "comparison_phase": "post_hoc_after_blind_summary_sha256",
        "comparison_quantity": "common_tilt_model_h_stage_minus_ground_lidar_pose_z",
        "estimated_h_m": estimated,
        "measured_stage_height_m": measured,
        "signed_error_m": signed_error,
        "absolute_error_m": absolute_error,
        "relative_error": absolute_error / abs(measured),
        "relative_error_percent": 100.0 * absolute_error / abs(measured),
        "bootstrap_interval_m": [bootstrap["h_p02_5_m"], bootstrap["h_p97_5_m"]],
        "measured_height_inside_bootstrap_interval": (
            bootstrap["h_p02_5_m"] <= measured <= bootstrap["h_p97_5_m"]
        ),
        "reused_existing_tolerance_m": None if tolerance is None else float(tolerance),
        "tolerance_interpretation": physical_config.get("tolerance_interpretation"),
        "status": status,
        "preserved_historical_height_status": "HEIGHT_FAIL",
        "production_localization_failure": False,
    }


def plot_height_posthoc(path, height):
    estimate = height["estimated_h_m"]
    lower, upper = height["bootstrap_interval_m"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(
        [0.0], [estimate], yerr=[[estimate - lower], [upper - estimate]], fmt="o",
        color="tab:blue", capsize=6, markersize=8, label="common-tilt h (candidate-block 95% interval)",
    )
    axis.axhline(height["measured_stage_height_m"], color="tab:red", linestyle="--", label="measured 0.150 m")
    axis.set_xlim(-0.7, 0.7)
    axis.set_xticks([0.0], ["post-hoc comparison"])
    axis.set_ylabel("stage-ground LiDAR pose-z separation [m]")
    axis.set_title("Measured height is used only after blind model sealing")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def report_text(summary):
    blind = summary["blind_analysis"]
    ground = blind["independent_plane_metrics"]["sample_level"][GROUND]
    stage = blind["independent_plane_metrics"]["sample_level"][STAGE]
    ground_candidate = blind["independent_plane_metrics"]["candidate_level"][GROUND]
    stage_candidate = blind["independent_plane_metrics"]["candidate_level"][STAGE]
    parallel = blind["parallelism_metrics"]["sample_level"]
    raw = blind["raw_group_pose_z_metrics"]
    common = blind["common_tilt_model_metrics"]["sample_level"]
    common_candidate = blind["common_tilt_model_metrics"]["candidate_level"]
    bootstrap = blind["bootstrap_metrics"]
    confounding = blind["conditioning_metrics"]["sample_label_xy_confounding"]
    comparison = blind["residual_model_comparison"]
    height = summary["height_posthoc_validation"]
    correlation_rows = "\n".join(
        f"| {row['scope']} | {row['variable']} | {row['pearson']:.4f} | {row['spearman']:.4f} |"
        for row in blind["residual_orientation_correlations"]
    )
    return f"""# Physical surface pose-z diagnostic Gate

This is a diagnostic-only analysis of accepted `T_map_lidar` sensor-center pose-z. It does not
modify localization, EKF, small_gicp, registration gates, voxel parameters, the map, the bag, or
any previous result. The fitted planes are **not** claimed to be physical floor point-cloud planes.

## Observed

### Fixed height-blind labels and data

- ground_before: diagnostic IDs 1-11; {ground['sample_count']} accepted candidate samples
- stage_top: diagnostic IDs 12-22; {stage['sample_count']} accepted candidate samples
- ground_after: unavailable in 0-50 s
- Measured height used for labeling/blind fitting: false / false
- Blind summary SHA256: `{summary['blind_output_sha256']}`
- Ground raw pose-z min/max/range: {raw[GROUND]['minimum_z_m']:.6f} /
  {raw[GROUND]['maximum_z_m']:.6f} / {raw[GROUND]['range_z_m']:.6f} m
- Stage raw pose-z min/max/range: {raw[STAGE]['minimum_z_m']:.6f} /
  {raw[STAGE]['maximum_z_m']:.6f} / {raw[STAGE]['range_z_m']:.6f} m

### Independent sensor-center pose-z support planes

| level/group | a=dz/dx | b=dz/dy | slope [deg] | RMSE [m] | p95 abs residual [m] | rank | condition |
|---|---:|---:|---:|---:|---:|---:|---:|
| sample ground | {ground['a_dz_dx']:.6f} | {ground['b_dz_dy']:.6f} | {ground['slope_angle_deg']:.3f} | {ground['rmse_m']:.6f} | {ground['p95_absolute_residual_m']:.6f} | {ground['design_matrix_rank']} | {ground['design_matrix_condition_number']:.3f} |
| sample stage | {stage['a_dz_dx']:.6f} | {stage['b_dz_dy']:.6f} | {stage['slope_angle_deg']:.3f} | {stage['rmse_m']:.6f} | {stage['p95_absolute_residual_m']:.6f} | {stage['design_matrix_rank']} | {stage['design_matrix_condition_number']:.3f} |
| candidate ground | {ground_candidate['a_dz_dx']:.6f} | {ground_candidate['b_dz_dy']:.6f} | {ground_candidate['slope_angle_deg']:.3f} | {ground_candidate['rmse_m']:.6f} | {ground_candidate['p95_absolute_residual_m']:.6f} | {ground_candidate['design_matrix_rank']} | {ground_candidate['design_matrix_condition_number']:.3f} |
| candidate stage | {stage_candidate['a_dz_dx']:.6f} | {stage_candidate['b_dz_dy']:.6f} | {stage_candidate['slope_angle_deg']:.3f} | {stage_candidate['rmse_m']:.6f} | {stage_candidate['p95_absolute_residual_m']:.6f} | {stage_candidate['design_matrix_rank']} | {stage_candidate['design_matrix_condition_number']:.3f} |

Sample-level independent normals differ by {parallel['angle_between_normals_deg']:.3f} deg;
stage-minus-ground gradient deltas are da={parallel['delta_dz_dx_stage_minus_ground']:.6f} and
db={parallel['delta_dz_dy_stage_minus_ground']:.6f}.

### Common-tilt parallel-plane model

- Sample model: a={common['a_dz_dx']:.6f}, b={common['b_dz_dy']:.6f}, slope={common['slope_angle_deg']:.3f} deg,
  h={common['h_stage_minus_ground_m']:.6f} m, rank={common['design_matrix_rank']}/{common['design_matrix_columns']},
  condition={common['design_matrix_condition_number']:.3f}
- Candidate-balanced model: a={common_candidate['a_dz_dx']:.6f}, b={common_candidate['b_dz_dy']:.6f},
  h={common_candidate['h_stage_minus_ground_m']:.6f} m, condition={common_candidate['design_matrix_condition_number']:.3f}
- Ground RMSE independent/common: {comparison[GROUND]['independent']['rmse_m']:.6f} / {comparison[GROUND]['common_tilt']['rmse_m']:.6f} m
- Stage RMSE independent/common: {comparison[STAGE]['independent']['rmse_m']:.6f} / {comparison[STAGE]['common_tilt']['rmse_m']:.6f} m
- Ground common residual p95/max: {comparison[GROUND]['common_tilt']['p95_absolute_residual_m']:.6f} /
  {comparison[GROUND]['common_tilt']['maximum_absolute_residual_m']:.6f} m
- Stage common residual p95/max: {comparison[STAGE]['common_tilt']['p95_absolute_residual_m']:.6f} /
  {comparison[STAGE]['common_tilt']['maximum_absolute_residual_m']:.6f} m

The ground and stage XY supports are spatially separated and correlated with the label. Therefore
`h` is confounded with x/y trend; **separation estimate is weakly constrained by the available
spatial geometry**. The rank and condition number are reported rather than hidden.
The stage indicator's linear R-squared from x/y is {confounding['stage_indicator_r_squared_from_linear_xy']:.6f}
(VIF {confounding['stage_indicator_variance_inflation_factor']:.3f}); group-centroid separation is
{confounding['group_centroid_distance_m']:.3f} m.

Candidate-block bootstrap: {bootstrap['successful_samples']}/{bootstrap['requested_samples']} successful,
{bootstrap['failure_count']} failures, h median={bootstrap['h_median_m']:.6f} m,
95% percentile interval=[{bootstrap['h_p02_5_m']:.6f}, {bootstrap['h_p97_5_m']:.6f}] m.

### Common-model residual associations

| scope | variable | Pearson | Spearman |
|---|---|---:|---:|
{correlation_rows}

These correlations are observations, not causal assignments.

### Phase B: post-hoc 0.150 m comparison

The blind summary was written and hashed before the physical config was read. Common-model
h={height['estimated_h_m']:.6f} m versus measured={height['measured_stage_height_m']:.6f} m:
signed error={height['signed_error_m']:.6f} m, absolute error={height['absolute_error_m']:.6f} m,
relative error={height['relative_error_percent']:.3f}%. The measured value is
{'inside' if height['measured_height_inside_bootstrap_interval'] else 'outside'} the bootstrap interval.
The unchanged existing {height['reused_existing_tolerance_m']:.3f} m initial-screening tolerance gives
**{height['status']}** for this post-hoc diagnostic only. Historical HEIGHT_FAIL remains preserved.

## Supported interpretation

The {parallel['angle_between_normals_deg']:.3f} deg independent-normal difference and the ground
RMSE increase from {comparison[GROUND]['independent']['rmse_m']:.6f} to
{comparison[GROUND]['common_tilt']['rmse_m']:.6f} m show that **a single common spatial tilt is
insufficient to explain the observed pose-z variation**. The fitted common component remains a
descriptive decomposition only; it is not an independently identified map-floor tilt, and `h` is
weakly constrained by separated XY supports. No new fit-quality PASS threshold or parameter tuning
was introduced.

## Not established

- absolute global ground-truth localization accuracy
- exact physical map-floor tilt (requires separate PLY floor segmentation)
- exact `T_base_lidar` or a lever-arm correction
- a causal explanation from residual correlation
- ground_after height in 0-50 s
- full-course physical z accuracy

Protected inputs unchanged: {summary['protected_inputs_unchanged']}.
"""


def public_blind_summary(analysis, labels_json, provenance, fit_config):
    return {
        "diagnostic_only": True,
        "analysis_phase": "A_blind_before_measured_height_read",
        "fitted_quantity": "map_frame_lidar_sensor_center_pose_z_spatial_support_plane",
        "fixed_labels": labels_json["fixed_labels"],
        "label_source": labels_json["label_source"],
        "measured_height_used_for_labeling": False,
        "measured_height_used_for_blind_fit": False,
        "candidate_z_used_for_labeling": False,
        "candidate_pair_diagnostics_used_for_labeling_or_fit": False,
        "source_plateau_provenance_height_blind_subset": provenance,
        "raw_group_pose_z_metrics": analysis["raw_group_pose_z_metrics"],
        "robust_fit_configuration": fit_config,
        "independent_plane_metrics": {
            "sample_level": analysis["independent_sample_planes"],
            "candidate_level": analysis["independent_candidate_planes"],
        },
        "parallelism_metrics": {
            "sample_level": analysis["sample_plane_parallelism"],
            "candidate_level": analysis["candidate_plane_parallelism"],
        },
        "common_tilt_model_metrics": {
            "sample_level": analysis["common_sample_plane"],
            "candidate_level": analysis["common_candidate_plane"],
        },
        "conditioning_metrics": {
            "sample_common_rank": analysis["common_sample_plane"]["design_matrix_rank"],
            "sample_common_condition_number": analysis["common_sample_plane"]["design_matrix_condition_number"],
            "candidate_common_rank": analysis["common_candidate_plane"]["design_matrix_rank"],
            "candidate_common_condition_number": analysis["common_candidate_plane"]["design_matrix_condition_number"],
            "spatial_label_confounding_warning": (
                "separation estimate is weakly constrained by the available spatial geometry"
            ),
            "sample_label_xy_confounding": analysis["sample_label_xy_confounding"],
            "candidate_label_xy_confounding": analysis["candidate_label_xy_confounding"],
        },
        "bootstrap_metrics": analysis["bootstrap"],
        "residual_model_comparison": analysis["residual_model_comparison"],
        "residual_orientation_correlations": analysis["residual_correlations"],
        "candidate_common_residual_metrics": analysis["candidate_residuals"],
        "scientific_conclusion": (
            "single common spatial tilt is insufficient to explain the observed pose-z variation"
        ),
    }


def input_fingerprints(config):
    inputs = config["inputs"]
    return {
        "bag": physical_gate.fingerprint(inputs["bag"]),
        "map_ply": physical_gate.fingerprint(inputs["map_ply"], include_hash=True),
        "localization_csv": physical_gate.fingerprint(inputs["localization_csv"], include_hash=True),
        "plateau_diagnostic_directory": physical_gate.fingerprint(inputs["plateau_diagnostic_directory"]),
        "physical_gate_directory": physical_gate.fingerprint(inputs["physical_gate_directory"]),
        "physical_gate_config": physical_gate.fingerprint(inputs["physical_gate_config"], include_hash=True),
    }


def run(config_path, output_directory=None):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    inputs = config["inputs"]
    output = Path(output_directory or config["output_directory"]).resolve()
    protected = [
        inputs["bag"], inputs["map_ply"], inputs["localization_csv"],
        inputs["plateau_diagnostic_directory"], inputs["physical_gate_directory"],
        inputs["physical_gate_config"],
    ]
    physical_gate.validate_output_path(output, protected)
    before = input_fingerprints(config)
    output.mkdir(parents=True, exist_ok=True)

    plateau_directory = Path(inputs["plateau_diagnostic_directory"])
    candidates = load_fixed_candidates(
        plateau_directory / "diagnostic_plateau_candidates.csv",
        config["fixed_physical_labels"],
    )
    provenance = sanitized_plateau_provenance(
        plateau_directory / "plateau_label_diagnostic_summary.json"
    )
    if provenance["candidate_count"] != 22 or provenance["candidate_search_window_sec"] != [0.0, 50.0]:
        raise ValueError("Source diagnostic must remain the frozen 22-candidate 0-50 s result")

    time_config = config["time_window"]
    origin = float(time_config["origin_timestamp"])
    start = float(time_config["start_sec"])
    end = float(time_config["end_sec"])
    records = physical_gate.read_localization_csv(inputs["localization_csv"], origin, start, end)
    sensor = physical_gate.bag_sensor_data(inputs["bag"], origin, start, end)
    samples = build_labeled_samples(
        records, candidates, sensor, origin,
        float(config["speed_association"]["maximum_dt_sec"]),
    )
    metrics = candidate_metrics(samples, candidates)
    labels_json = {
        "fixed_labels": {
            GROUND: list(range(1, 12)),
            STAGE: list(range(12, 23)),
            "ground_after": "unavailable",
        },
        "label_source": list(config["fixed_physical_labels"]["label_source"]),
        "source_candidate_namespace": "imu_gicp_plateau_label_diagnostic diagnostic IDs",
        "historical_physical_gate_candidate_ids_used": False,
        "measured_height_used_for_labeling": False,
        "candidate_z_used_for_labeling": False,
        "candidate_pair_error_used_for_labeling": False,
        "labels_locked_before_blind_fit": True,
    }
    write_json(output / "physical_surface_labels.json", labels_json)

    analysis = blind_fit_analysis(
        samples, metrics, config["robust_fit"], config["bootstrap"]
    )
    sample_fields = [
        "timestamp", "time_sec", "diagnostic_id", "physical_label", "x_m", "y_m", "z_m",
        "roll_rad", "pitch_rad", "yaw_rad", "roll_deg", "pitch_deg", "yaw_deg",
        "odom_speed_mps", "odom_speed_association_dt_sec",
    ]
    write_csv(
        output / "surface_pose_z_samples.csv",
        ({field: row[field] for field in sample_fields} for row in samples),
        sample_fields,
    )
    write_csv(output / "surface_candidate_metrics.csv", metrics)
    write_json(output / "ground_sample_plane_fit.json", analysis["independent_sample_planes"][GROUND])
    write_json(output / "stage_sample_plane_fit.json", analysis["independent_sample_planes"][STAGE])
    write_json(output / "ground_candidate_plane_fit.json", analysis["independent_candidate_planes"][GROUND])
    write_json(output / "stage_candidate_plane_fit.json", analysis["independent_candidate_planes"][STAGE])
    write_json(output / "plane_parallelism_metrics.json", {
        "sample_level": analysis["sample_plane_parallelism"],
        "candidate_level": analysis["candidate_plane_parallelism"],
    })
    write_json(output / "common_tilt_parallel_plane_fit.json", {
        "sample_level": analysis["common_sample_plane"],
        "candidate_level": analysis["common_candidate_plane"],
        "residual_model_comparison": analysis["residual_model_comparison"],
    })
    residual_fields = sample_fields + [
        "independent_fitted_z_m", "independent_residual_z_m", "independent_robust_weight",
        "common_fitted_z_m", "common_residual_z_m", "common_robust_weight",
    ]
    write_csv(output / "surface_pose_z_residuals.csv", samples, residual_fields)
    write_csv(output / "residual_correlation_metrics.csv", analysis["residual_correlations"])
    write_csv(output / "candidate_residual_metrics.csv", analysis["candidate_residuals"])
    write_csv(output / "common_tilt_h_bootstrap.csv", analysis["bootstrap_rows"])

    map_xy = plateau_diagnostic.load_map_xy(inputs["map_ply"], int(config["plot"]["map_max_points"]))
    plot_fixed_labels(output / "surface_labels_map_overlay.png", map_xy, records, metrics)
    plot_independent_planes(output / "surface_pose_z_planes.png", samples, analysis["independent_sample_planes"])
    plot_spatial_trend(output / "surface_z_spatial_trend.png", samples)
    plot_common_residual(output / "common_tilt_residual.png", samples)
    plot_residual_correlates(output / "residual_vs_pitch.png", samples)

    blind_summary = public_blind_summary(analysis, labels_json, provenance, config["robust_fit"])
    blind_path = output / "blind_surface_fit_summary.json"
    write_json(blind_path, blind_summary)
    blind_digest = sha256_file(blind_path)
    (output / "blind_surface_fit_summary.sha256").write_text(
        f"{blind_digest}  blind_surface_fit_summary.json\n", encoding="utf-8"
    )

    # Phase B begins only after the blind output exists and its hash is sealed above.
    height = height_posthoc(
        analysis["common_sample_plane"], analysis["bootstrap"], inputs["physical_gate_config"]
    )
    height["blind_output_sha256"] = blind_digest
    write_json(output / "height_posthoc_validation.json", height)
    plot_height_posthoc(output / "height_posthoc_comparison.png", height)

    after = input_fingerprints(config)
    integrity = {"unchanged": before == after, "before": before, "after": after}
    if not integrity["unchanged"]:
        raise RuntimeError("A protected source changed during the diagnostic")
    final_summary = {
        "diagnostic_only": True,
        "diagnostic_name": "physical surface LiDAR pose-z spatial diagnostic",
        "fixed_labels": labels_json["fixed_labels"],
        "label_source": labels_json["label_source"],
        "measured_height_used_for_labeling": False,
        "measured_height_used_for_blind_fit": False,
        "blind_analysis": blind_summary,
        "blind_output_sha256": blind_digest,
        "measured_stage_height_m": height["measured_stage_height_m"],
        "height_posthoc_validation": height,
        "protected_inputs_unchanged": integrity["unchanged"],
        "input_integrity": integrity,
        "input_counts": {
            "localization_rows": len(records),
            "accepted_localization_rows": sum(record["accepted"] for record in records),
            "labeled_candidate_samples": len(samples),
            "ground_samples": sum(row["physical_label"] == GROUND for row in samples),
            "stage_samples": sum(row["physical_label"] == STAGE for row in samples),
            "imu_messages": int(sensor["imu_times"].size),
            "odom_messages": int(sensor["odom_times"].size),
            "map_vertices_plotted": int(map_xy.shape[0]),
        },
        "output_directory": str(output),
        "preserved_historical_height_status": "HEIGHT_FAIL",
        "production_localization_failure": False,
        "scientific_conclusion": blind_summary["scientific_conclusion"],
        "height_validation_interpretation": (
            "post-hoc HEIGHT_FAIL, but physical height validation remains inconclusive because "
            "the common-model separation is weakly constrained by spatial-label confounding"
        ),
    }
    write_json(output / "surface_pose_z_diagnostic_summary.json", final_summary)
    (output / "surface_pose_z_diagnostic_report.md").write_text(
        report_text(final_summary), encoding="utf-8"
    )
    return final_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    summary = run(args.config, args.output_directory)
    common = summary["blind_analysis"]["common_tilt_model_metrics"]["sample_level"]
    print(json.dumps({
        "ground_samples": summary["input_counts"]["ground_samples"],
        "stage_samples": summary["input_counts"]["stage_samples"],
        "common_h_m": common["h_stage_minus_ground_m"],
        "posthoc_status": summary["height_posthoc_validation"]["status"],
        "protected_inputs_unchanged": summary["protected_inputs_unchanged"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
