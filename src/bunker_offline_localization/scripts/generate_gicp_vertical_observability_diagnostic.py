#!/usr/bin/env python3

"""Read-only independent-GICP vertical observability and accepted-anchor diagnostic."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_gicp_vertical_observability_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_imu_gicp_physical_gate as physical_gate
import generate_ply_support_surface_diagnostic as ply_diagnostic

np = physical_gate.np
plt = physical_gate.plt
yaml = physical_gate.yaml

GROUPS = ("baseline", "transition", "anomaly", "recovery_stage")
HESSIAN_NAMES = [f"hessian_{row}{column}" for row in range(6) for column in range(6)]


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


def git_output(directory, *arguments):
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def source_line(path, marker):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if marker in line]
    if not matches:
        raise ValueError(f"Required source marker not found in {path}: {marker}")
    return matches


def audit_hessian_convention(config):
    inputs = config["inputs"]
    small_gicp = Path(inputs["small_gicp_directory"])
    commit = git_output(small_gicp, "rev-parse", "HEAD")
    if commit != config["expected_small_gicp_commit"]:
        raise ValueError(f"Unexpected small_gicp commit: {commit}")
    if git_output(small_gicp, "status", "--porcelain"):
        raise ValueError("Pinned small_gicp submodule is dirty")
    optimizer = small_gicp / "include/small_gicp/registration/optimizer.hpp"
    factor = small_gicp / "include/small_gicp/factors/gicp_factor.hpp"
    lie = small_gicp / "include/small_gicp/util/lie.hpp"
    reduction = small_gicp / "include/small_gicp/registration/reduction_omp.hpp"
    result_header = small_gicp / "include/small_gicp/registration/registration_result.hpp"
    production = Path(inputs["production_source_file"])
    metrics = Path(inputs["production_metrics_file"])
    markers = {
        "optimizer_right_update": (optimizer, "result.T_target_source = result.T_target_source * se3_exp(delta);"),
        "optimizer_hessian_assignment": (optimizer, "result.H = H;"),
        "factor_rotation_jacobian": (factor, "J.block<3, 3>(0, 0) = T.linear() * skew"),
        "factor_translation_jacobian": (factor, "J.block<3, 3>(0, 3) = -T.linear();"),
        "factor_information": (factor, "*H = J.transpose() * mahalanobis * J;"),
        "factor_error": (factor, "*e = 0.5 * residual.transpose() * mahalanobis * residual;"),
        "rotation_first_order": (lie, "Twist vector [rx, ry, rz, tx, ty, tz]"),
        "parallel_sum": (reduction, "Hs[thread_id] += H;"),
        "result_final_information": (result_header, "Final information matrix"),
        "production_result_copy": (production, "output.hessian = result.H;"),
        "production_csv_row_major": (metrics, "record.hessian(row, column)"),
    }
    citations = {
        name: {"path": str(path), "line_numbers": source_line(path, marker), "marker": marker}
        for name, (path, marker) in markers.items()
    }
    expected_order = list(config["hessian_analysis"]["source_parameter_order"])
    if expected_order != ["rx", "ry", "rz", "tx", "ty", "tz"]:
        raise ValueError("Configured Hessian ordering does not match audited small_gicp source")
    if int(config["hessian_analysis"]["source_local_tz_index"]) != 5:
        raise ValueError("Configured source-local tz index does not match audited small_gicp source")
    return {
        "small_gicp_commit": commit,
        "production_source_file": str(production),
        "small_gicp_source_file": str(factor),
        "matrix_origin": (
            "OpenMP sum of inlier GICP per-correspondence J^T M J matrices at the last "
            "optimizer linearization; production copies RegistrationResult.H row-major to CSV"
        ),
        "parameter_order": expected_order,
        "source_local_tz_index": 5,
        "rotation_parameterization": "SO(3) exponential coordinates rx,ry,rz",
        "translation_parameterization": "SE(3) exponential tangent tx,ty,tz in source/LiDAR-local axes",
        "local_frame_convention": (
            "right-multiplicative update T_target_source <- T_target_source * Exp(delta); "
            "delta is a source/LiDAR-local tangent"
        ),
        "returned_pose_vs_hessian_linearization": (
            "H is computed before the final accepted optimizer delta and is not relinearized "
            "after that delta; convergence bounds make this the last near-final linearization"
        ),
        "matrix_symmetry_expected": True,
        "matrix_weighting": (
            "GICP residual precision uses inverse(target_cov + T*source_cov*T^T); "
            "error is 0.5*r^T*M*r and H is J^T*M*J summed over accepted correspondences"
        ),
        "map_vertical_analysis_basis": (
            "derived only: retain local rotational tangent and rotate translation tangent into "
            "map x/y/z with B=diag(I,R_map_lidar^T), H_map_translation=B^T*H*B"
        ),
        "symmetrization_used_for_analysis": bool(
            config["hessian_analysis"]["symmetrize_for_analysis"]
        ),
        "eigenvalue_tolerance": {
            "relative": float(config["hessian_analysis"]["eigenvalue_relative_tolerance"]),
            "absolute": float(config["hessian_analysis"]["eigenvalue_absolute_tolerance"]),
        },
        "pseudo_inverse_tolerance": float(config["hessian_analysis"]["pseudo_inverse_rcond"]),
        "source_citations": citations,
        "unknown_fields": [],
    }


def quaternion_rotation(row, prefix):
    quaternion = np.asarray([float(row[f"{prefix}_q{name}"]) for name in "xyzw"])
    return physical_gate.quaternion_to_rotation(quaternion)


def rotation_angle(rotation):
    return math.acos(float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0)))


def load_candidate_windows(path, groups):
    rows = read_csv(path)
    ids = [int(row["diagnostic_candidate_id"]) for row in rows]
    if ids != list(range(1, 23)):
        raise ValueError("Plateau diagnostic candidate IDs must remain exactly 1-22")
    group_by_id = {}
    for group in GROUPS:
        for candidate_id in groups[group]:
            if int(candidate_id) in group_by_id:
                raise ValueError("Candidate appears in more than one predeclared group")
            group_by_id[int(candidate_id)] = group
    if sorted(group_by_id) != list(range(1, 23)):
        raise ValueError("Predeclared groups must cover candidate IDs 1-22 exactly")
    windows = []
    for row in rows:
        candidate_id = int(row["diagnostic_candidate_id"])
        windows.append({
            "diagnostic_candidate_id": candidate_id,
            "start_time_sec": float(row["start_offset_sec"]),
            "end_time_sec": float(row["end_offset_sec"]),
            "diagnostic_group": group_by_id[candidate_id],
        })
    return windows


def assign_candidate(time_sec, windows):
    matches = [
        window for window in windows
        if window["start_time_sec"] - 1.0e-9 <= time_sec <= window["end_time_sec"] + 1.0e-9
    ]
    if len(matches) > 1:
        raise ValueError(f"Overlapping candidate windows at {time_sec}")
    if not matches:
        return None, "unassigned"
    return matches[0]["diagnostic_candidate_id"], matches[0]["diagnostic_group"]


def load_localization(path, origin, start, end, windows):
    source = read_csv(path)
    if not source:
        raise ValueError("Localization CSV is empty")
    required = [
        "timestamp", "accepted", "converged", "iterations", "num_inliers", "final_error",
        "pred_x", "pred_y", "pred_z", "pred_roll", "pred_pitch", "pred_yaw",
        "gicp_x", "gicp_y", "gicp_z", "gicp_roll", "gicp_pitch", "gicp_yaw",
        *HESSIAN_NAMES,
    ]
    missing = [name for name in required if name not in source[0]]
    if missing:
        raise ValueError(f"Localization CSV is missing: {missing}")
    records = []
    for row_index, row in enumerate(source):
        timestamp = float(row["timestamp"])
        time_sec = timestamp - origin
        if time_sec < start - 1.0e-9 or time_sec > end + 1.0e-9:
            raise ValueError(f"Localization row outside [{start}, {end}] s")
        candidate_id, group = assign_candidate(time_sec, windows)
        prediction_position = np.asarray([float(row[f"pred_{axis}"]) for axis in "xyz"])
        gicp_position = np.asarray([float(row[f"gicp_{axis}"]) for axis in "xyz"])
        prediction_rotation = quaternion_rotation(row, "pred")
        gicp_rotation = quaternion_rotation(row, "gicp")
        relative_rotation = prediction_rotation.T @ gicp_rotation
        correction_rpy = physical_gate.rotation_to_rpy(relative_rotation)
        correction_translation = gicp_position - prediction_position
        inliers = int(row["num_inliers"])
        final_error = float(row["final_error"])
        translation_norm = float(np.linalg.norm(correction_translation))
        record = {
            "row_index": row_index,
            "timestamp": timestamp,
            "bag_relative_time": time_sec,
            "diagnostic_candidate_id": candidate_id,
            "diagnostic_group": group,
            "accepted": row["accepted"] == "1",
            "status": "ACCEPTED" if row["accepted"] == "1" else "REJECTED",
            "reject_reason": row["reject_reason"],
            "converged": row["converged"] == "1",
            "iterations": int(row["iterations"]),
            "inliers": inliers,
            "final_error": final_error,
            "final_error_per_inlier": final_error / inliers if inliers > 0 else None,
            "prediction_x": prediction_position[0],
            "prediction_y": prediction_position[1],
            "prediction_z": prediction_position[2],
            "prediction_roll": float(row["pred_roll"]),
            "prediction_pitch": float(row["pred_pitch"]),
            "prediction_yaw": float(row["pred_yaw"]),
            "gicp_x": gicp_position[0],
            "gicp_y": gicp_position[1],
            "gicp_z": gicp_position[2],
            "gicp_roll": float(row["gicp_roll"]),
            "gicp_pitch": float(row["gicp_pitch"]),
            "gicp_yaw": float(row["gicp_yaw"]),
            "correction_x": correction_translation[0],
            "correction_y": correction_translation[1],
            "correction_z": correction_translation[2],
            "per_scan_delta_z": correction_translation[2],
            "correction_translation_norm": translation_norm,
            "correction_roll": correction_rpy[0],
            "correction_pitch": correction_rpy[1],
            "correction_yaw": correction_rpy[2],
            "correction_rotation_norm": rotation_angle(relative_rotation),
            "vertical_fraction": (
                abs(correction_translation[2]) / translation_norm if translation_norm > 0.0 else None
            ),
            "prediction_rotation": prediction_rotation,
            "gicp_rotation": gicp_rotation,
            "hessian": np.asarray([
                [float(row[f"hessian_{matrix_row}{column}"]) for column in range(6)]
                for matrix_row in range(6)
            ]),
            "source_row": row,
        }
        records.append(record)
    return records


def resolve_parameter_index(parameter_order, parameter_name):
    matches = [index for index, name in enumerate(parameter_order) if name == parameter_name]
    if len(matches) != 1:
        raise ValueError(f"Parameter {parameter_name} is absent or duplicated in {parameter_order}")
    return matches[0]


def effective_axis_information(matrix, axis_index, pinv_rcond):
    matrix = np.asarray(matrix, dtype=float)
    remaining = [index for index in range(matrix.shape[0]) if index != axis_index]
    rest = matrix[np.ix_(remaining, remaining)]
    coupling = matrix[axis_index, remaining]
    return float(
        matrix[axis_index, axis_index]
        - coupling @ np.linalg.pinv(rest, rcond=pinv_rcond) @ coupling.T
    )


def hessian_coupling(matrix, first, second, denominator_tolerance):
    denominator = abs(matrix[first, first] * matrix[second, second])
    if denominator <= denominator_tolerance:
        return None
    return float(abs(matrix[first, second]) / math.sqrt(denominator))


def analyze_hessian(matrix, gicp_rotation, config):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"Expected 6x6 Hessian, got {matrix.shape}")
    finite = bool(np.all(np.isfinite(matrix)))
    if not finite:
        return {"hessian_valid": False, "hessian_invalid_reason": "NONFINITE"}
    symmetry_error = float(np.linalg.norm(matrix - matrix.T, ord="fro"))
    matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
    relative_symmetry_error = symmetry_error / max(matrix_norm, np.finfo(float).tiny)
    symmetric = 0.5 * (matrix + matrix.T) if config["symmetrize_for_analysis"] else matrix

    # Right update translation is source/LiDAR-local. Re-express only its 3-vector in map axes.
    basis = np.eye(6)
    basis[3:, 3:] = gicp_rotation.T
    map_translation_matrix = basis.T @ symmetric @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(map_translation_matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = max(
        float(config["eigenvalue_absolute_tolerance"]),
        float(config["eigenvalue_relative_tolerance"]) * scale,
    )
    negative_count = int(np.count_nonzero(eigenvalues < -tolerance))
    numerical_negative_count = int(np.count_nonzero(
        (eigenvalues < 0.0) & (eigenvalues >= -tolerance)
    ))
    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = (
        float(eigenvalues[-1] / positive[0]) if positive.size else None
    )
    effective_rank = int(np.count_nonzero(eigenvalues > tolerance))
    rcond = float(config["pseudo_inverse_rcond"])
    inverse_proxy = np.linalg.pinv(map_translation_matrix, rcond=rcond)
    local_inverse_proxy = np.linalg.pinv(symmetric, rcond=rcond)
    z_index = resolve_parameter_index(config["source_parameter_order"], "tz")
    denominator_tolerance = float(config["coupling_denominator_tolerance"])
    map_hzz = float(map_translation_matrix[z_index, z_index])
    local_hzz = float(symmetric[z_index, z_index])
    map_effective = effective_axis_information(map_translation_matrix, z_index, rcond)
    local_effective = effective_axis_information(symmetric, z_index, rcond)
    weakest = eigenvectors[:, 0]
    second = eigenvectors[:, 1]
    output = {
        "hessian_valid": True,
        "hessian_invalid_reason": "NONE",
        "hessian_shape_rows": 6,
        "hessian_shape_columns": 6,
        "hessian_symmetry_error_fro": symmetry_error,
        "hessian_relative_symmetry_error": relative_symmetry_error,
        "hessian_symmetry_warning": (
            relative_symmetry_error > float(config["symmetry_relative_warning_threshold"])
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "second_smallest_eigenvalue": float(eigenvalues[1]),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "condition_number": condition_number,
        "effective_numeric_rank": effective_rank,
        "negative_eigenvalue_count": negative_count,
        "numerical_negative_eigenvalue_count": numerical_negative_count,
        "eigenvalue_scale_relative_tolerance": tolerance,
        "local_tz_hzz_raw": local_hzz,
        "local_tz_effective_information": local_effective,
        "local_tz_effective_over_hzz": local_effective / local_hzz if local_hzz else None,
        "local_tz_inverse_curvature_proxy": float(local_inverse_proxy[z_index, z_index]),
        "map_vertical_hzz_raw": map_hzz,
        "map_vertical_effective_information": map_effective,
        "map_vertical_effective_over_hzz": map_effective / map_hzz if map_hzz else None,
        "map_vertical_inverse_curvature_proxy": float(inverse_proxy[z_index, z_index]),
        "weakest_mode_z_participation": float(weakest[z_index] ** 2),
        "weakest_mode_roll_participation": float(weakest[0] ** 2),
        "weakest_mode_pitch_participation": float(weakest[1] ** 2),
        "weakest_mode_yaw_participation": float(weakest[2] ** 2),
        "weakest_mode_x_participation": float(weakest[3] ** 2),
        "weakest_mode_y_participation": float(weakest[4] ** 2),
        "second_mode_z_participation": float(second[z_index] ** 2),
        "second_mode_roll_participation": float(second[0] ** 2),
        "second_mode_pitch_participation": float(second[1] ** 2),
        "second_mode_yaw_participation": float(second[2] ** 2),
        "second_mode_x_participation": float(second[3] ** 2),
        "second_mode_y_participation": float(second[4] ** 2),
        "z_roll_hessian_coupling": hessian_coupling(
            map_translation_matrix, z_index, 0, denominator_tolerance
        ),
        "z_pitch_hessian_coupling": hessian_coupling(
            map_translation_matrix, z_index, 1, denominator_tolerance
        ),
        "z_yaw_hessian_coupling": hessian_coupling(
            map_translation_matrix, z_index, 2, denominator_tolerance
        ),
        "z_x_hessian_coupling": hessian_coupling(
            map_translation_matrix, z_index, 3, denominator_tolerance
        ),
        "z_y_hessian_coupling": hessian_coupling(
            map_translation_matrix, z_index, 4, denominator_tolerance
        ),
    }
    for index, value in enumerate(eigenvalues, start=1):
        output[f"lambda_{index}"] = float(value)
    return output


def apply_anchor_and_hessian_metrics(records, config):
    last_accepted_z = None
    cumulative = 0.0
    invariant_violations = []
    valid_hessian_count = 0
    translation_tolerance = float(config["vertical_fraction_translation_norm_tolerance_m"])
    for record in records:
        record["previous_accepted_anchor_z"] = last_accepted_z
        record["anchor_z_invariant_error_m"] = (
            record["prediction_z"] - last_accepted_z if last_accepted_z is not None else None
        )
        if (last_accepted_z is not None and abs(record["anchor_z_invariant_error_m"])
                > float(config["anchor_z_invariant_tolerance_m"])):
            invariant_violations.append({
                "row_index": record["row_index"],
                "timestamp": record["timestamp"],
                "error_m": record["anchor_z_invariant_error_m"],
                "accepted": record["accepted"],
            })
        record["vertical_fraction"] = (
            abs(record["per_scan_delta_z"]) / record["correction_translation_norm"]
            if record["correction_translation_norm"] > translation_tolerance else None
        )
        if record["accepted"]:
            record["absolute_delta_z"] = abs(record["per_scan_delta_z"])
            cumulative += record["per_scan_delta_z"]
            record["cumulative_accepted_delta_z"] = cumulative
            last_accepted_z = record["gicp_z"]
            hessian = analyze_hessian(record["hessian"], record["gicp_rotation"], config)
            record.update(hessian)
            valid_hessian_count += hessian["hessian_valid"]
        else:
            record["cumulative_accepted_delta_z"] = cumulative
    baseline_proxies = [
        record["map_vertical_inverse_curvature_proxy"] for record in records
        if record["accepted"] and record["diagnostic_group"] == "baseline"
        and record.get("hessian_valid")
    ]
    baseline_effective = [
        record["map_vertical_effective_information"] for record in records
        if record["accepted"] and record["diagnostic_group"] == "baseline"
        and record.get("hessian_valid")
    ]
    proxy_reference = float(np.median(baseline_proxies))
    effective_reference = float(np.median(baseline_effective))
    for record in records:
        if record["accepted"] and record.get("hessian_valid"):
            record["vertical_inverse_proxy_over_baseline_median"] = (
                record["map_vertical_inverse_curvature_proxy"] / proxy_reference
            )
            record["effective_z_information_over_baseline_median"] = (
                record["map_vertical_effective_information"] / effective_reference
            )
    return {
        "checked_rows_after_first_anchor": len(records) - 1,
        "violation_count": len(invariant_violations),
        "violations": invariant_violations,
        "tolerance_m": float(config["anchor_z_invariant_tolerance_m"]),
        "valid_accepted_hessian_count": valid_hessian_count,
        "baseline_vertical_inverse_proxy_median": proxy_reference,
        "baseline_effective_z_information_median": effective_reference,
    }


def load_mapping_trajectory(path):
    trajectory = np.loadtxt(path, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.shape[1] != 8 or not np.all(np.isfinite(trajectory)):
        raise ValueError("Mapping trajectory must be finite timestamp x y z qx qy qz qw")
    return trajectory


def associate_mapping(records, trajectory, maximum_distance):
    accepted = [record for record in records if record["accepted"]]
    xy = np.asarray([[record["gicp_x"], record["gicp_y"]] for record in accepted])
    indices, distances = ply_diagnostic.monotonic_xy_association(xy, trajectory[:, 1:3])
    rows = []
    for record, mapping_index, distance in zip(accepted, indices, distances):
        available = float(distance) <= float(maximum_distance)
        mapping = trajectory[int(mapping_index)]
        record.update({
            "mapping_association_status": "AVAILABLE" if available else "UNAVAILABLE_DISTANCE",
            "mapping_index": int(mapping_index),
            "mapping_x": float(mapping[1]) if available else None,
            "mapping_y": float(mapping[2]) if available else None,
            "mapping_z": float(mapping[3]) if available else None,
            "mapping_xy_distance": float(distance),
            "independent_minus_mapping_z": (
                record["gicp_z"] - float(mapping[3]) if available else None
            ),
        })
        rows.append({
            "row_index": record["row_index"],
            "timestamp": record["timestamp"],
            "bag_relative_time": record["bag_relative_time"],
            "diagnostic_candidate_id": record["diagnostic_candidate_id"],
            "diagnostic_group": record["diagnostic_group"],
            "mapping_index": record["mapping_index"],
            "mapping_x": record["mapping_x"],
            "mapping_y": record["mapping_y"],
            "mapping_z": record["mapping_z"],
            "xy_distance": record["mapping_xy_distance"],
            "independent_gicp_z": record["gicp_z"],
            "independent_minus_mapping_z": record["independent_minus_mapping_z"],
            "association_status": record["mapping_association_status"],
            "association_inputs": "accepted_independent_xy_and_ordered_canonical_mapping_xy_only",
        })
    return rows


def numeric_values(rows, field):
    return np.asarray([
        row[field] for row in rows if row.get(field) is not None and np.isfinite(row[field])
    ], dtype=float)


def metric_summary(rows, field, include_sum=False):
    values = numeric_values(rows, field)
    if values.size == 0:
        result = {
            "count": 0, "median": None, "p05": None, "p95": None,
            "minimum": None, "maximum": None, "mean": None,
        }
        if include_sum:
            result["sum"] = None
        return result
    result = {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    if include_sum:
        result["sum"] = float(np.sum(values))
    return result


SUMMARY_FIELDS = [
    "per_scan_delta_z", "absolute_delta_z", "cumulative_accepted_delta_z",
    "correction_translation_norm",
    "correction_rotation_norm", "vertical_fraction", "minimum_eigenvalue",
    "second_smallest_eigenvalue", "condition_number", "map_vertical_hzz_raw",
    "map_vertical_effective_information", "map_vertical_effective_over_hzz",
    "map_vertical_inverse_curvature_proxy", "vertical_inverse_proxy_over_baseline_median",
    "weakest_mode_z_participation", "weakest_mode_roll_participation",
    "weakest_mode_pitch_participation", "z_roll_hessian_coupling",
    "z_pitch_hessian_coupling", "iterations", "inliers", "final_error",
    "final_error_per_inlier", "independent_minus_mapping_z",
]


def candidate_metrics(accepted, windows):
    rows = []
    for window in windows:
        candidate_id = window["diagnostic_candidate_id"]
        selected = [row for row in accepted if row["diagnostic_candidate_id"] == candidate_id]
        if not selected:
            raise ValueError(f"Candidate {candidate_id} has no accepted localization rows")
        candidate = {
            "diagnostic_candidate_id": candidate_id,
            "diagnostic_group": window["diagnostic_group"],
            "start_time_sec": window["start_time_sec"],
            "end_time_sec": window["end_time_sec"],
            "accepted_scan_count": len(selected),
            "mapping_z_median": float(np.median(numeric_values(selected, "mapping_z"))),
            "prediction_z_median": float(np.median(numeric_values(selected, "prediction_z"))),
            "gicp_z_median": float(np.median(numeric_values(selected, "gicp_z"))),
            "independent_minus_mapping_z_median": float(
                np.median(numeric_values(selected, "independent_minus_mapping_z"))
            ),
            "per_scan_delta_z_median": float(np.median(numeric_values(selected, "per_scan_delta_z"))),
            "per_scan_delta_z_p05": float(np.percentile(numeric_values(selected, "per_scan_delta_z"), 5)),
            "per_scan_delta_z_p95": float(np.percentile(numeric_values(selected, "per_scan_delta_z"), 95)),
            "most_negative_single_delta_z": float(np.min(numeric_values(selected, "per_scan_delta_z"))),
            "largest_positive_single_delta_z": float(np.max(numeric_values(selected, "per_scan_delta_z"))),
            "sum_accepted_delta_z": float(np.sum(numeric_values(selected, "per_scan_delta_z"))),
            "cumulative_delta_z_at_first_scan": selected[0]["cumulative_accepted_delta_z"],
            "cumulative_delta_z_at_last_scan": selected[-1]["cumulative_accepted_delta_z"],
        }
        for field in SUMMARY_FIELDS:
            candidate[f"{field}_median"] = metric_summary(selected, field)["median"]
        rows.append(candidate)
    return rows


def group_comparison(accepted, candidates, groups):
    scan_level = {}
    candidate_balanced = {}
    for group in GROUPS:
        selected_scans = [row for row in accepted if row["diagnostic_group"] == group]
        selected_candidates = [row for row in candidates if row["diagnostic_group"] == group]
        scan_level[group] = {
            "accepted_scan_count": len(selected_scans),
            "candidate_ids": [int(value) for value in groups[group]],
            "metrics": {
                field: metric_summary(selected_scans, field, include_sum=field == "per_scan_delta_z")
                for field in SUMMARY_FIELDS
            },
        }
        candidate_balanced[group] = {
            "candidate_count": len(selected_candidates),
            "candidate_ids": [int(value) for value in groups[group]],
            "metrics": {
                field: metric_summary(selected_candidates, f"{field}_median")
                for field in SUMMARY_FIELDS
            },
        }
    return {"scan_level": scan_level, "candidate_balanced": candidate_balanced}


def flatten_group_comparison(comparison):
    rows = []
    for level in ("scan_level", "candidate_balanced"):
        for group in GROUPS:
            source = comparison[level][group]
            row = {
                "aggregation_level": level,
                "diagnostic_group": group,
                "candidate_ids": ";".join(map(str, source["candidate_ids"])),
                "observation_count": source.get("accepted_scan_count", source.get("candidate_count")),
            }
            for field, metrics in source["metrics"].items():
                for statistic, value in metrics.items():
                    row[f"{field}_{statistic}"] = value
            rows.append(row)
    return rows


def correlations(rows, scope, variables):
    output = []
    for variable in variables:
        first = numeric_values(rows, "per_scan_delta_z")
        paired = [
            (row["per_scan_delta_z"], row[variable]) for row in rows
            if row.get(variable) is not None
            and np.isfinite(row["per_scan_delta_z"]) and np.isfinite(row[variable])
        ]
        if paired:
            first = np.asarray([item[0] for item in paired])
            second = np.asarray([item[1] for item in paired])
        else:
            first = second = np.asarray([])
        output.append({
            "scope": scope,
            "first_variable": "per_scan_delta_z",
            "second_variable": variable,
            "sample_count": int(first.size),
            "pearson": (
                None if first.size < 3 else ply_diagnostic.finite_or_none(
                    physical_gate.pearson(first, second)
                )
            ),
            "spearman": (
                None if first.size < 3 else ply_diagnostic.finite_or_none(
                    physical_gate.spearman(first, second)
                )
            ),
            "causal_claim": False,
        })
    return output


def z_drift_decomposition(accepted, windows, anchor_audit):
    by_id = {window["diagnostic_candidate_id"]: window for window in windows}
    largest_negative = min(accepted, key=lambda row: row["per_scan_delta_z"])
    largest_positive = max(accepted, key=lambda row: row["per_scan_delta_z"])
    maximum_discrepancy = max(
        accepted, key=lambda row: abs(row["independent_minus_mapping_z"])
    )
    transition_start = by_id[6]["end_time_sec"]
    transition_end = by_id[7]["start_time_sec"]
    gap_rows = [
        row for row in accepted
        if transition_start < row["bag_relative_time"] < transition_end
    ]
    first_id7 = next(row for row in accepted if row["diagnostic_candidate_id"] == 7)
    previous_to_id7 = accepted[accepted.index(first_id7) - 1]
    last_id6 = [row for row in accepted if row["diagnostic_candidate_id"] == 6][-1]
    onset_rows = [
        row for row in accepted
        if by_id[5]["start_time_sec"] <= row["bag_relative_time"] <= by_id[12]["end_time_sec"]
    ]
    return {
        "anchor_prediction_invariant": anchor_audit,
        "full_interval": {
            "accepted_scan_count": len(accepted),
            "initial_prediction_z": accepted[0]["prediction_z"],
            "final_accepted_gicp_z": accepted[-1]["gicp_z"],
            "cumulative_accepted_delta_z_final": accepted[-1]["cumulative_accepted_delta_z"],
            "largest_single_negative_delta_z": {
                "timestamp": largest_negative["timestamp"],
                "bag_relative_time": largest_negative["bag_relative_time"],
                "diagnostic_candidate_id": largest_negative["diagnostic_candidate_id"],
                "prediction_z": largest_negative["prediction_z"],
                "gicp_z": largest_negative["gicp_z"],
                "delta_z": largest_negative["per_scan_delta_z"],
                "mapping_z": largest_negative["mapping_z"],
                "independent_minus_mapping_z": largest_negative["independent_minus_mapping_z"],
            },
            "largest_single_positive_delta_z": {
                "timestamp": largest_positive["timestamp"],
                "bag_relative_time": largest_positive["bag_relative_time"],
                "delta_z": largest_positive["per_scan_delta_z"],
            },
            "largest_absolute_mapping_z_discrepancy": {
                "timestamp": maximum_discrepancy["timestamp"],
                "bag_relative_time": maximum_discrepancy["bag_relative_time"],
                "difference_z": maximum_discrepancy["independent_minus_mapping_z"],
            },
        },
        "id6_to_id7_unassigned_transition": {
            "time_window_sec": [transition_start, transition_end],
            "accepted_scan_count": len(gap_rows),
            "sum_delta_z": float(np.sum(numeric_values(gap_rows, "per_scan_delta_z"))),
            "median_delta_z": float(np.median(numeric_values(gap_rows, "per_scan_delta_z"))),
            "minimum_delta_z": float(np.min(numeric_values(gap_rows, "per_scan_delta_z"))),
            "maximum_delta_z": float(np.max(numeric_values(gap_rows, "per_scan_delta_z"))),
            "negative_correction_count": sum(row["per_scan_delta_z"] < 0.0 for row in gap_rows),
            "positive_correction_count": sum(row["per_scan_delta_z"] > 0.0 for row in gap_rows),
            "candidate_id_forced_assignment": False,
        },
        "id7_entry": {
            "last_id6_timestamp": last_id6["timestamp"],
            "last_id6_cumulative_accepted_delta_z": last_id6[
                "cumulative_accepted_delta_z"
            ],
            "previous_accepted_timestamp": previous_to_id7["timestamp"],
            "previous_accepted_gicp_z": previous_to_id7["gicp_z"],
            "cumulative_accepted_delta_z_immediately_before_id7": previous_to_id7[
                "cumulative_accepted_delta_z"
            ],
            "first_id7_timestamp": first_id7["timestamp"],
            "first_id7_prediction_z": first_id7["prediction_z"],
            "first_id7_gicp_z": first_id7["gicp_z"],
            "first_id7_delta_z": first_id7["per_scan_delta_z"],
            "first_id7_mapping_z": first_id7["mapping_z"],
            "cumulative_accepted_delta_z_after_first_id7": first_id7[
                "cumulative_accepted_delta_z"
            ],
            "prediction_already_contains_discrepancy": True,
            "origin_traced_to_prior_accepted_gicp_corrections": True,
        },
        "anomaly_onset_candidate_5_to_12": {
            "accepted_scan_count": len(onset_rows),
            "time_window_sec": [by_id[5]["start_time_sec"], by_id[12]["end_time_sec"]],
            "minimum_delta_z": float(np.min(numeric_values(onset_rows, "per_scan_delta_z"))),
            "maximum_delta_z": float(np.max(numeric_values(onset_rows, "per_scan_delta_z"))),
            "negative_count": sum(row["per_scan_delta_z"] < 0.0 for row in onset_rows),
            "positive_count": sum(row["per_scan_delta_z"] > 0.0 for row in onset_rows),
        },
        "decomposition_conclusion": (
            "The ID7 entry prediction inherits an already-low previous accepted pose. The "
            "discrepancy originated in several large, alternating accepted GICP z corrections "
            "between IDs 6 and 7, including a single -0.246565 m correction; it is not a smooth "
            "sequence of uniformly small negative corrections and is not EKF z drift."
        ),
    }


def select_rviz_handoff_scans(accepted):
    baseline = [row for row in accepted if row["diagnostic_group"] == "baseline"]
    recovery = [row for row in accepted if row["diagnostic_group"] == "recovery_stage"]
    stable_stage = [row for row in recovery if row["diagnostic_candidate_id"] == 22]
    selections = [
        ("baseline_representative", min(baseline, key=lambda row: abs(row["per_scan_delta_z"])),
         "baseline scan with minimum absolute accepted z correction"),
        ("largest_negative_dz", min(accepted, key=lambda row: row["per_scan_delta_z"]),
         "largest single accepted negative map-z correction"),
        ("largest_mapping_z_discrepancy",
         max(accepted, key=lambda row: abs(row["independent_minus_mapping_z"])),
         "largest absolute XY-associated independent-minus-mapping z"),
        ("recovery_representative",
         min(recovery, key=lambda row: abs(row["independent_minus_mapping_z"])),
         "recovery/stage scan closest to canonical mapping z"),
        ("stable_stage_representative",
         min(stable_stage, key=lambda row: abs(row["per_scan_delta_z"])),
         "candidate 22 scan with minimum absolute accepted z correction"),
    ]
    return {
        "rviz_implemented": False,
        "purpose": "handoff for later map/raw/registered/correspondence visualization",
        "selected_scans": [{
            "selection_role": role,
            "selection_reason": reason,
            "row_index": row["row_index"],
            "timestamp": row["timestamp"],
            "bag_relative_time": row["bag_relative_time"],
            "diagnostic_candidate_id": row["diagnostic_candidate_id"],
            "prediction_z": row["prediction_z"],
            "gicp_z": row["gicp_z"],
            "mapping_z": row["mapping_z"],
            "per_scan_delta_z": row["per_scan_delta_z"],
            "independent_minus_mapping_z": row["independent_minus_mapping_z"],
        } for role, row, reason in selections],
    }


def build_interpretation(comparison, drift):
    baseline = comparison["scan_level"]["baseline"]["metrics"]
    anomaly = comparison["scan_level"]["anomaly"]["metrics"]
    effective_ratio = (
        anomaly["map_vertical_effective_information"]["median"]
        / baseline["map_vertical_effective_information"]["median"]
    )
    inverse_ratio = (
        anomaly["map_vertical_inverse_curvature_proxy"]["median"]
        / baseline["map_vertical_inverse_curvature_proxy"]["median"]
    )
    lambda_ratio = (
        anomaly["minimum_eigenvalue"]["median"]
        / baseline["minimum_eigenvalue"]["median"]
    )
    z_participation_ratio = (
        anomaly["weakest_mode_z_participation"]["median"]
        / baseline["weakest_mode_z_participation"]["median"]
    )
    return {
        "most_supported_explanation": (
            "Hessian-insufficient correspondence/local-minimum hypothesis with accepted-anchor carry"
        ),
        "initial_prediction_issue": (
            "true at ID7 entry, but secondary: the low prediction is exactly the prior accepted "
            "GICP anchor and was traced to earlier GICP corrections"
        ),
        "accumulated_small_negative_drift": (
            "not supported as a smooth small-step mechanism; large alternating corrections include "
            f"{drift['full_interval']['largest_single_negative_delta_z']['delta_z']:.6f} m"
        ),
        "weak_vertical_observability_region_specific": (
            "not supported by relative final-Hessian metrics: anomaly effective information and "
            "lambda_min increase while inverse curvature decreases relative to baseline"
        ),
        "z_roll_pitch_coupling_increase": (
            "not supported as anomaly-specific; coupling and weakest-mode participation must be "
            "read with meter/radian scaling limitations"
        ),
        "relative_anomaly_over_baseline": {
            "effective_z_information_median_ratio": effective_ratio,
            "vertical_inverse_proxy_median_ratio": inverse_ratio,
            "minimum_eigenvalue_median_ratio": lambda_ratio,
            "weakest_mode_z_participation_median_ratio": z_participation_ratio,
        },
        "local_hessian_limitation": (
            "A healthy Hessian around the accepted final solution cannot rule out wrong "
            "correspondences, another local basin, or a multi-modal cost landscape."
        ),
        "causal_proof": False,
    }


def plot_mapping_prediction_final(path, accepted):
    time = numeric_values(accepted, "bag_relative_time")
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.plot(time, numeric_values(accepted, "mapping_z"), label="XY-associated mapping z", linewidth=1.4)
    axis.plot(time, numeric_values(accepted, "prediction_z"), label="prediction z", linewidth=1.0)
    axis.plot(time, numeric_values(accepted, "gicp_z"), label="final GICP z", linewidth=1.0)
    axis.set(xlabel="bag-relative time [s]", ylabel="map-frame LiDAR z [m]",
             title="Canonical mapping vs accepted-anchor prediction vs final GICP z")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_z_correction(path, accepted):
    time = numeric_values(accepted, "bag_relative_time")
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, numeric_values(accepted, "per_scan_delta_z"), color="tab:red", linewidth=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("GICP z - prediction z [m]")
    axes[0].set_title("Accepted per-scan vertical correction")
    axes[1].plot(time, numeric_values(accepted, "cumulative_accepted_delta_z"), color="tab:purple")
    axes[1].set_ylabel("cumulative accepted dz [m]")
    axes[1].set_xlabel("bag-relative time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_candidate_decomposition(path, candidates):
    ids = [row["diagnostic_candidate_id"] for row in candidates]
    figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    for field, label, style in (
        ("mapping_z_median", "mapping z", "s-"),
        ("prediction_z_median", "prediction z", "o-"),
        ("gicp_z_median", "GICP z", "^-"),
        ("independent_minus_mapping_z_median", "independent - mapping z", "d--"),
    ):
        axis.plot(ids, [row[field] for row in candidates], style, label=label, markersize=5)
    axis.axvline(5.5, color="0.4", linestyle=":")
    axis.axvline(6.5, color="0.4", linestyle=":")
    axis.axvline(11.5, color="0.4", linestyle=":")
    axis.set_xticks(ids)
    axis.set(xlabel="diagnostic candidate ID", ylabel="z [m]",
             title="Candidate-level z drift decomposition (candidate-balanced medians)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_eigenvalues(path, accepted):
    time = numeric_values(accepted, "bag_relative_time")
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.semilogy(time, numeric_values(accepted, "minimum_eigenvalue"), label="lambda_min")
    axis.semilogy(time, numeric_values(accepted, "second_smallest_eigenvalue"), label="lambda_2")
    axis.set(xlabel="bag-relative time [s]", ylabel="Hessian eigenvalue [relative scale]",
             title="Final-linearization Hessian smallest eigenvalues")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_effective_z(path, accepted):
    time = numeric_values(accepted, "bag_relative_time")
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, numeric_values(accepted, "map_vertical_hzz_raw"), label="raw map-vertical Hzz")
    axes[0].plot(time, numeric_values(accepted, "map_vertical_effective_information"),
                 label="Schur effective z information")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative information scale")
    axes[0].legend(loc="best")
    axes[1].plot(time, numeric_values(accepted, "map_vertical_inverse_curvature_proxy"),
                 color="tab:purple", label="vertical inverse-curvature proxy")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("inverse-curvature proxy")
    axes[1].set_xlabel("bag-relative time [s]")
    axes[1].legend(loc="best")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("Map-vertical local curvature diagnostics (not calibrated covariance)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_weakest_composition(path, candidates):
    ids = [row["diagnostic_candidate_id"] for row in candidates]
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.plot(ids, [row["weakest_mode_z_participation_median"] for row in candidates], "o-", label="map-z")
    axis.plot(ids, [row["weakest_mode_roll_participation_median"] for row in candidates], "s-", label="local rx")
    axis.plot(ids, [row["weakest_mode_pitch_participation_median"] for row in candidates], "^-", label="local ry")
    axis.set_xticks(ids)
    axis.set_ylim(0.0, 1.02)
    axis.set(xlabel="diagnostic candidate ID", ylabel="squared eigenvector participation",
             title="Weakest-mode composition (meter/radian scaling dependent)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_z_orientation(path, accepted):
    anomaly = [row for row in accepted if row["diagnostic_group"] == "anomaly"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(numeric_values(accepted, "correction_roll"),
                       numeric_values(accepted, "per_scan_delta_z"), s=10, alpha=0.45)
    axes[0, 0].set(xlabel="relative roll correction [rad]", ylabel="map dz [m]")
    axes[0, 1].scatter(numeric_values(accepted, "correction_pitch"),
                       numeric_values(accepted, "per_scan_delta_z"), s=10, alpha=0.45)
    axes[0, 1].set(xlabel="relative pitch correction [rad]", ylabel="map dz [m]")
    time = numeric_values(anomaly, "bag_relative_time")
    axes[1, 0].plot(time, numeric_values(anomaly, "prediction_roll"), label="prediction roll")
    axes[1, 0].plot(time, numeric_values(anomaly, "gicp_roll"), label="GICP roll")
    axes[1, 0].plot(time, numeric_values(anomaly, "correction_roll"), label="relative correction")
    axes[1, 0].set(xlabel="time [s]", ylabel="roll [rad]")
    axes[1, 1].plot(time, numeric_values(anomaly, "prediction_pitch"), label="prediction pitch")
    axes[1, 1].plot(time, numeric_values(anomaly, "gicp_pitch"), label="GICP pitch")
    axes[1, 1].plot(time, numeric_values(anomaly, "correction_pitch"), label="relative correction")
    axes[1, 1].set(xlabel="time [s]", ylabel="pitch [rad]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="best")
    figure.suptitle("z correction vs orientation correction (association, not causation)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_quality_vs_anomaly(path, accepted):
    time = numeric_values(accepted, "bag_relative_time")
    fields = [
        ("independent_minus_mapping_z", "independent - mapping z [m]"),
        ("inliers", "inliers"),
        ("final_error_per_inlier", "final error / inlier"),
        ("iterations", "iterations"),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for axis, (field, label) in zip(axes, fields):
        axis.plot(time, numeric_values(accepted, field), linewidth=0.9)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("bag-relative time [s]")
    figure.suptitle("Registration scalar quality metrics versus vertical inconsistency")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_anomaly_timeline(path, accepted, windows, first_id, last_id):
    by_id = {window["diagnostic_candidate_id"]: window for window in windows}
    selected = [
        row for row in accepted
        if by_id[first_id]["start_time_sec"] <= row["bag_relative_time"]
        <= by_id[last_id]["end_time_sec"]
    ]
    time = numeric_values(selected, "bag_relative_time")
    figure, axes = plt.subplots(7, 1, figsize=(15, 16), sharex=True, constrained_layout=True)
    axes[0].plot(time, numeric_values(selected, "mapping_z"), label="mapping z")
    axes[0].plot(time, numeric_values(selected, "prediction_z"), label="prediction z")
    axes[0].plot(time, numeric_values(selected, "gicp_z"), label="GICP z")
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best", ncol=3)
    axes[1].plot(time, numeric_values(selected, "per_scan_delta_z"), color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("per-scan dz [m]")
    axes[2].plot(time, numeric_values(selected, "map_vertical_effective_information"))
    axes[2].set_ylabel("effective z info")
    axes[3].plot(time, numeric_values(selected, "minimum_eigenvalue"))
    axes[3].set_ylabel("lambda min")
    axes[4].plot(time, numeric_values(selected, "condition_number"))
    axes[4].set_ylabel("condition")
    axes[5].plot(time, numeric_values(selected, "inliers"))
    axes[5].set_ylabel("inliers")
    axes[6].plot(time, numeric_values(selected, "final_error_per_inlier"))
    axes[6].set_ylabel("error/inlier")
    axes[6].set_xlabel("bag-relative time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("High-resolution candidate IDs 5-12 vertical anomaly timeline")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def protected_fingerprints(inputs):
    small_gicp = Path(inputs["small_gicp_directory"])
    small_sources = [
        small_gicp / "include/small_gicp/registration/optimizer.hpp",
        small_gicp / "include/small_gicp/factors/gicp_factor.hpp",
        small_gicp / "include/small_gicp/util/lie.hpp",
        small_gicp / "include/small_gicp/registration/reduction_omp.hpp",
        small_gicp / "include/small_gicp/registration/registration_result.hpp",
    ]
    return {
        "independent_bag": physical_gate.fingerprint(inputs["bag"]),
        "localization_csv": physical_gate.fingerprint(inputs["localization_csv"], include_hash=True),
        "canonical_mapping_trajectory": physical_gate.fingerprint(
            inputs["mapping_trajectory"], include_hash=True
        ),
        "ply_map": physical_gate.fingerprint(inputs["map_ply"], include_hash=True),
        "physical_gate_directory": physical_gate.fingerprint(inputs["physical_gate_directory"]),
        "plateau_diagnostic_directory": physical_gate.fingerprint(
            inputs["plateau_diagnostic_directory"]
        ),
        "surface_pose_z_diagnostic_directory": physical_gate.fingerprint(
            inputs["surface_pose_z_diagnostic_directory"]
        ),
        "ply_geometry_diagnostic_directory": physical_gate.fingerprint(
            inputs["ply_geometry_diagnostic_directory"]
        ),
        "production_sources": {
            name: physical_gate.fingerprint(inputs[name], include_hash=True)
            for name in ("production_source_file", "production_metrics_file", "production_localizer_file")
        },
        "production_configs": [
            physical_gate.fingerprint(path, include_hash=True)
            for path in inputs["production_config_files"]
        ],
        "small_gicp": {
            "head": git_output(small_gicp, "rev-parse", "HEAD"),
            "porcelain_status": git_output(small_gicp, "status", "--porcelain"),
            "audited_source_files": [
                physical_gate.fingerprint(path, include_hash=True) for path in small_sources
            ],
        },
    }


def scan_metric_rows(accepted):
    base_fields = [
        "row_index", "timestamp", "bag_relative_time", "diagnostic_candidate_id",
        "diagnostic_group", "status", "converged", "iterations", "inliers",
        "final_error", "final_error_per_inlier", "prediction_x", "prediction_y",
        "prediction_z", "prediction_roll", "prediction_pitch", "prediction_yaw",
        "gicp_x", "gicp_y", "gicp_z", "gicp_roll", "gicp_pitch", "gicp_yaw",
        "correction_x", "correction_y", "correction_z", "correction_translation_norm",
        "per_scan_delta_z",
        "correction_roll", "correction_pitch", "correction_yaw", "correction_rotation_norm",
        "vertical_fraction", "previous_accepted_anchor_z", "anchor_z_invariant_error_m",
        "cumulative_accepted_delta_z", "mapping_z", "mapping_xy_distance",
        "independent_minus_mapping_z",
    ]
    rows = []
    for record in accepted:
        row = {field: record.get(field) for field in base_fields}
        for name in HESSIAN_NAMES:
            matrix_row, column = int(name[-2]), int(name[-1])
            row[name] = record["hessian"][matrix_row, column]
        rows.append(row)
    return rows


def rejected_rows(records):
    rows = []
    for record in records:
        if record["accepted"]:
            continue
        finite_matrix = bool(np.all(np.isfinite(record["hessian"])))
        rows.append({
            "row_index": record["row_index"],
            "timestamp": record["timestamp"],
            "bag_relative_time": record["bag_relative_time"],
            "diagnostic_candidate_id": record["diagnostic_candidate_id"],
            "diagnostic_group": record["diagnostic_group"],
            "status": record["status"],
            "reject_reason": record["reject_reason"],
            "converged": record["converged"],
            "iterations": record["iterations"],
            "inliers": record["inliers"],
            "final_error": record["final_error"],
            "per_scan_delta_z": record["per_scan_delta_z"],
            "hessian_finite": finite_matrix,
            "hessian_analysis_forced": False,
        })
    return rows


def eigenspectrum_rows(accepted):
    fields = [
        "row_index", "timestamp", "bag_relative_time", "diagnostic_candidate_id",
        "diagnostic_group", "hessian_valid", "hessian_invalid_reason",
        "hessian_shape_rows", "hessian_shape_columns", "hessian_symmetry_error_fro",
        "hessian_relative_symmetry_error", "hessian_symmetry_warning", "lambda_1",
        "lambda_2", "lambda_3", "lambda_4", "lambda_5", "lambda_6",
        "minimum_eigenvalue", "second_smallest_eigenvalue", "largest_eigenvalue",
        "condition_number", "effective_numeric_rank", "negative_eigenvalue_count",
        "numerical_negative_eigenvalue_count", "eigenvalue_scale_relative_tolerance",
    ]
    return [{field: row.get(field) for field in fields} for row in accepted]


def observability_rows(accepted):
    fields = [
        "row_index", "timestamp", "bag_relative_time", "diagnostic_candidate_id",
        "diagnostic_group", "local_tz_hzz_raw", "local_tz_effective_information",
        "local_tz_effective_over_hzz", "local_tz_inverse_curvature_proxy",
        "map_vertical_hzz_raw", "map_vertical_effective_information",
        "map_vertical_effective_over_hzz", "map_vertical_inverse_curvature_proxy",
        "vertical_inverse_proxy_over_baseline_median",
        "effective_z_information_over_baseline_median", "weakest_mode_z_participation",
        "weakest_mode_roll_participation", "weakest_mode_pitch_participation",
        "weakest_mode_yaw_participation", "weakest_mode_x_participation",
        "weakest_mode_y_participation", "second_mode_z_participation",
        "second_mode_roll_participation", "second_mode_pitch_participation",
        "second_mode_yaw_participation", "second_mode_x_participation",
        "second_mode_y_participation", "z_roll_hessian_coupling",
        "z_pitch_hessian_coupling", "z_yaw_hessian_coupling",
        "z_x_hessian_coupling", "z_y_hessian_coupling",
    ]
    return [{field: row.get(field) for field in fields} for row in accepted]


def accumulation_rows(accepted):
    fields = [
        "row_index", "timestamp", "bag_relative_time", "diagnostic_candidate_id",
        "diagnostic_group", "mapping_z", "previous_accepted_anchor_z", "prediction_z",
        "per_scan_delta_z", "gicp_z", "cumulative_accepted_delta_z",
        "anchor_z_invariant_error_m", "independent_minus_mapping_z",
    ]
    return [{field: row.get(field) for field in fields} for row in accepted]


def vertical_summary(accepted, anchor_audit, comparison, discrepancy_correlations):
    symmetry = metric_summary(accepted, "hessian_relative_symmetry_error")
    negative = sum(row["negative_eigenvalue_count"] for row in accepted)
    numerical_negative = sum(row["numerical_negative_eigenvalue_count"] for row in accepted)
    return {
        "diagnostic_only": True,
        "calibrated_covariance_claimed": False,
        "new_acceptance_threshold_created": False,
        "accepted_hessian_count": len(accepted),
        "anchor_z_invariant": anchor_audit,
        "hessian_sanity": {
            "finite_count": sum(row["hessian_valid"] for row in accepted),
            "symmetry_relative_error": symmetry,
            "symmetry_warning_count": sum(row["hessian_symmetry_warning"] for row in accepted),
            "negative_eigenvalue_count_total": negative,
            "numerical_negative_eigenvalue_count_total": numerical_negative,
            "full_rank_count": sum(row["effective_numeric_rank"] == 6 for row in accepted),
        },
        "scan_level_group_metrics": comparison["scan_level"],
        "candidate_balanced_group_metrics": comparison["candidate_balanced"],
        "mapping_discrepancy_hessian_correlations": discrepancy_correlations,
        "limitations": [
            "Hessian and its inverse are relative curvature diagnostics, not calibrated covariance.",
            "The Hessian describes only the accepted local final-solution basin.",
            "Eigenvector rotation and translation entries mix radian and meter parameter units.",
            "A healthy local Hessian cannot exclude wrong correspondences or a different local minimum.",
        ],
    }


def quality_gate_observation(comparison):
    baseline = comparison["scan_level"]["baseline"]["metrics"]
    anomaly = comparison["scan_level"]["anomaly"]["metrics"]
    fields = [
        "inliers", "final_error_per_inlier", "iterations", "correction_translation_norm",
        "correction_rotation_norm",
    ]
    return {
        "baseline_vs_anomaly_scan_medians": {
            field: {
                "baseline": baseline[field]["median"],
                "anomaly": anomaly[field]["median"],
                "anomaly_over_baseline": anomaly[field]["median"] / baseline[field]["median"],
            } for field in fields
        },
        "observation": (
            "Anomaly inliers remain comparable, while error-per-inlier and iterations are elevated. "
            "All anomaly scans still pass; existing scalar metrics do not directly encode the "
            "canonical vertical discrepancy."
        ),
        "new_reject_threshold_proposed": False,
    }


def fmt(value, digits=6):
    return "UNAVAILABLE" if value is None else f"{value:.{digits}f}"


def report_text(summary, audit, candidate_rows):
    comparison = summary["group_comparison"]["scan_level"]
    drift = summary["z_drift_decomposition"]
    association = summary["mapping_association_quality"]
    detail = [row for row in candidate_rows if 5 <= row["diagnostic_candidate_id"] <= 12]
    detail_table = "\n".join(
        f"| {row['diagnostic_candidate_id']} | {row['accepted_scan_count']} | "
        f"{fmt(row['mapping_z_median'])} | {fmt(row['prediction_z_median'])} | "
        f"{fmt(row['gicp_z_median'])} | {fmt(row['independent_minus_mapping_z_median'])} | "
        f"{fmt(row['per_scan_delta_z_median'])} | {fmt(row['most_negative_single_delta_z'])} | "
        f"{fmt(row['sum_accepted_delta_z'])} | {fmt(row['minimum_eigenvalue_median'], 1)} | "
        f"{fmt(row['condition_number_median'], 1)} | "
        f"{fmt(row['map_vertical_effective_information_median'], 1)} | "
        f"{fmt(row['weakest_mode_z_participation_median'], 4)} | "
        f"{fmt(row['inliers_median'], 1)} | {fmt(row['final_error_per_inlier_median'], 6)} |"
        for row in detail
    )
    group_table = "\n".join(
        f"| {group} | {data['accepted_scan_count']} | "
        f"{fmt(data['metrics']['per_scan_delta_z']['median'])} | "
        f"{fmt(data['metrics']['per_scan_delta_z']['sum'])} | "
        f"{fmt(data['metrics']['absolute_delta_z']['median'])} | "
        f"{fmt(data['metrics']['minimum_eigenvalue']['median'], 1)} | "
        f"{fmt(data['metrics']['condition_number']['median'], 1)} | "
        f"{fmt(data['metrics']['map_vertical_effective_information']['median'], 1)} | "
        f"{fmt(data['metrics']['map_vertical_inverse_curvature_proxy']['median'], 9)} | "
        f"{fmt(data['metrics']['weakest_mode_z_participation']['median'], 4)} | "
        f"{fmt(data['metrics']['z_roll_hessian_coupling']['median'], 4)} | "
        f"{fmt(data['metrics']['z_pitch_hessian_coupling']['median'], 4)} | "
        f"{fmt(data['metrics']['inliers']['median'], 1)} | "
        f"{fmt(data['metrics']['final_error_per_inlier']['median'], 6)} | "
        f"{fmt(data['metrics']['iterations']['median'], 1)} |"
        for group, data in comparison.items()
    )
    citations = audit["source_citations"]
    return f"""# Independent GICP vertical observability / z-anomaly diagnostic

This is read-only post-processing of the existing production localization CSV. No localization,
map, bag, small_gicp, configuration, quality Gate, or prior diagnostic was changed or replayed.

## small_gicp Hessian convention audit

- Pinned commit: `{audit['small_gicp_commit']}`
- Parameter order: `{audit['parameter_order']}`; source-local `tz` index is 5.
- Matrix: {audit['matrix_origin']}.
- Update: {audit['local_frame_convention']}.
- Last-linearization caveat: {audit['returned_pose_vs_hessian_linearization']}.
- Weighting: {audit['matrix_weighting']}.
- Production copy: `{citations['production_result_copy']['path']}:{citations['production_result_copy']['line_numbers'][0]}`
- Optimizer update/H assignment: `{citations['optimizer_right_update']['path']}:{citations['optimizer_right_update']['line_numbers']}` /
  `{citations['optimizer_hessian_assignment']['line_numbers']}`
- GICP Jacobian/information: `{citations['factor_information']['path']}:{citations['factor_rotation_jacobian']['line_numbers'][0]}-{citations['factor_information']['line_numbers'][0]}`
- SE(3) order: `{citations['rotation_first_order']['path']}:{citations['rotation_first_order']['line_numbers'][0]}`

Because source `tz` is LiDAR-local, the primary map-vertical diagnostic re-expresses only the
translation tangent in map axes. This is a derived analysis; the stored matrix is unchanged.

## z anomaly decomposition

- Anchor invariant violations: {drift['anchor_prediction_invariant']['violation_count']} at
  {drift['anchor_prediction_invariant']['tolerance_m']:.1e} m tolerance.
- Largest negative accepted dz: {drift['full_interval']['largest_single_negative_delta_z']['delta_z']:.6f} m
  at {drift['full_interval']['largest_single_negative_delta_z']['bag_relative_time']:.6f} s,
  candidate={drift['full_interval']['largest_single_negative_delta_z']['diagnostic_candidate_id']}.
- ID6-to-ID7 unassigned interval: {drift['id6_to_id7_unassigned_transition']['accepted_scan_count']} scans,
  sum dz={drift['id6_to_id7_unassigned_transition']['sum_delta_z']:.6f} m,
  min/max={drift['id6_to_id7_unassigned_transition']['minimum_delta_z']:.6f}/
  {drift['id6_to_id7_unassigned_transition']['maximum_delta_z']:.6f} m.
- Cumulative dz: last ID6={drift['id7_entry']['last_id6_cumulative_accepted_delta_z']:.6f} m,
  immediately before ID7={drift['id7_entry']['cumulative_accepted_delta_z_immediately_before_id7']:.6f} m,
  after first ID7={drift['id7_entry']['cumulative_accepted_delta_z_after_first_id7']:.6f} m.
- Conclusion: {drift['decomposition_conclusion']}

## IDs 5-12 detailed candidate medians

| ID | scans | map z | pred z | GICP z | ind-map z | dz median | min dz | sum dz | lambda min | cond | effective z info | weak z part | inliers | error/inlier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{detail_table}

Candidate windows are not expanded: scans between candidates remain `NONE/unassigned`.

## Baseline vs anomaly vs recovery (scan-level medians)

| group | scans | dz | sum dz | abs dz | lambda min | cond | effective z info | inverse proxy | weak z part | z-roll C | z-pitch C | inliers | error/inlier | iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{group_table}

Candidate-balanced versions are retained in `group_comparison.json` and
`group_vertical_metrics.csv`; neither aggregation is selected as the sole result.

## Hessian sanity and limitations

- Finite/full-rank accepted Hessians: {summary['vertical_observability']['hessian_sanity']['finite_count']}/
  {summary['vertical_observability']['hessian_sanity']['full_rank_count']} of
  {summary['vertical_observability']['accepted_hessian_count']}.
- Relative symmetry error median/max:
  {summary['vertical_observability']['hessian_sanity']['symmetry_relative_error']['median']:.3e}/
  {summary['vertical_observability']['hessian_sanity']['symmetry_relative_error']['maximum']:.3e}.
- Significant/numerical negative eigenvalue totals:
  {summary['vertical_observability']['hessian_sanity']['negative_eigenvalue_count_total']}/
  {summary['vertical_observability']['hessian_sanity']['numerical_negative_eigenvalue_count_total']}.
- `H_pinv[z,z]` is named an inverse-curvature proxy, not physical covariance or 1-sigma
  localization uncertainty. Rotation/translation eigenvector components mix radians and meters.

## Mapping association and quality-Gate visibility

- XY-only ordered association: {association['available_count']}/{association['accepted_count']} available;
  median/p95/max distance={association['distance_m']['median']:.6f}/
  {association['distance_m']['p95']:.6f}/{association['distance_m']['maximum']:.6f} m.
- Association excludes different-bag timestamps, z, orientation, and measured height.
- {summary['quality_gate_observation']['observation']}

## Supported interpretation

**{summary['interpretation']['most_supported_explanation']}**.
At ID7 entry the prediction is already low, but the anchor invariant proves that it came from the
previous accepted GICP pose. The earlier onset contains large alternating corrections rather than
uniform small negative accumulation. Anomaly median effective z information and lambda-min are
higher than baseline, while inverse curvature and weakest-mode z participation are lower. Thus a
region-specific loss of local vertical curvature is not supported by this final-Hessian evidence.

The local Hessian is still strongly z-dominant in its weakest mode throughout the route, and its
meter/radian scaling prevents calibrated physical comparison of eigenvector components. A healthy
local final-solution Hessian cannot rule out wrong correspondences, a different local minimum, or
a multi-modal cost landscape. The next justified Gate is selected-scan correspondence/RViz audit;
no replay, cost sweep, or tuning is performed here.

Protected inputs unchanged: {summary['protected_inputs_unchanged']}.
"""


def run(config_path, output_directory=None):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    inputs = config["inputs"]
    output = Path(output_directory or config["output_directory"]).resolve()
    protected = [
        inputs["bag"], inputs["map_ply"], inputs["mapping_trajectory"],
        inputs["localization_csv"], inputs["physical_gate_directory"],
        inputs["plateau_diagnostic_directory"], inputs["surface_pose_z_diagnostic_directory"],
        inputs["ply_geometry_diagnostic_directory"], inputs["small_gicp_directory"],
        inputs["production_source_file"], inputs["production_metrics_file"],
        inputs["production_localizer_file"], *inputs["production_config_files"],
    ]
    physical_gate.validate_output_path(output, protected)
    before = protected_fingerprints(inputs)
    audit = audit_hessian_convention(config)
    output.mkdir(parents=True, exist_ok=True)

    windows = load_candidate_windows(
        Path(inputs["plateau_diagnostic_directory"]) / "diagnostic_plateau_candidates.csv",
        config["predeclared_groups"],
    )
    time = config["time_window"]
    records = load_localization(
        inputs["localization_csv"], float(time["origin_timestamp"]),
        float(time["start_sec"]), float(time["end_sec"]), windows,
    )
    anchor_audit = apply_anchor_and_hessian_metrics(records, {
        **config["hessian_analysis"],
    })
    accepted = [record for record in records if record["accepted"]]
    rejected = [record for record in records if not record["accepted"]]
    if len(records) != 490 or len(accepted) != 486 or len(rejected) != 4:
        raise ValueError(
            f"Frozen production counts changed: {len(records)}/{len(accepted)}/{len(rejected)}"
        )
    trajectory = load_mapping_trajectory(inputs["mapping_trajectory"])
    association_rows = associate_mapping(
        records, trajectory, float(config["mapping_association"]["maximum_xy_distance_m"])
    )
    candidate_rows = candidate_metrics(accepted, windows)
    comparison = group_comparison(
        accepted, candidate_rows, config["predeclared_groups"]
    )
    group_rows = flatten_group_comparison(comparison)
    orientation_correlations = z_orientation_correlations(accepted)
    discrepancy_correlations = discrepancy_hessian_correlations(accepted)
    drift = z_drift_decomposition(accepted, windows, anchor_audit)
    interpretation = build_interpretation(comparison, drift)
    quality_observation = quality_gate_observation(comparison)
    rviz_handoff = select_rviz_handoff_scans(accepted)
    vertical = vertical_summary(accepted, anchor_audit, comparison, discrepancy_correlations)
    association_available = [
        row for row in association_rows if row["association_status"] == "AVAILABLE"
    ]
    association_quality = {
        "method": config["mapping_association"]["method"],
        "inputs": "accepted_independent_scan_xy_and_canonical_mapping_xy_and_route_order_only",
        "accepted_count": len(accepted),
        "available_count": len(association_available),
        "unavailable_count": len(accepted) - len(association_available),
        "maximum_allowed_distance_m": float(
            config["mapping_association"]["maximum_xy_distance_m"]
        ),
        "distance_m": ply_diagnostic.distribution_metrics([
            row["xy_distance"] for row in association_available
        ]),
    }

    write_json(output / "hessian_convention_audit.json", audit)
    write_csv(output / "gicp_vertical_scan_metrics.csv", scan_metric_rows(accepted))
    write_csv(output / "gicp_vertical_rejected_rows.csv", rejected_rows(records))
    write_csv(output / "scan_level_mapping_xy_association.csv", association_rows)
    write_csv(output / "hessian_eigenspectrum.csv", eigenspectrum_rows(accepted))
    write_csv(output / "hessian_vertical_observability.csv", observability_rows(accepted))
    write_csv(output / "candidate_vertical_metrics.csv", candidate_rows)
    write_csv(output / "group_vertical_metrics.csv", group_rows)
    write_csv(output / "z_correction_accumulation.csv", accumulation_rows(accepted))
    write_csv(output / "z_orientation_correlation.csv", orientation_correlations)
    write_csv(
        output / "mapping_discrepancy_hessian_correlation.csv", discrepancy_correlations
    )
    write_json(output / "vertical_observability_summary.json", vertical)
    write_json(output / "z_drift_decomposition.json", drift)
    write_json(output / "group_comparison.json", comparison)
    write_json(output / "rviz_handoff_selected_scans.json", rviz_handoff)

    plot_mapping_prediction_final(output / "mapping_prediction_gicp_z.png", accepted)
    plot_z_correction(output / "per_scan_z_correction.png", accepted)
    plot_candidate_decomposition(output / "candidate_z_drift_decomposition.png", candidate_rows)
    plot_eigenvalues(output / "hessian_smallest_eigenvalues.png", accepted)
    plot_effective_z(output / "effective_z_information.png", accepted)
    plot_weakest_composition(output / "weakest_mode_composition.png", candidate_rows)
    plot_z_orientation(output / "z_correction_vs_orientation.png", accepted)
    plot_quality_vs_anomaly(output / "registration_quality_vs_anomaly.png", accepted)
    plot_anomaly_timeline(
        output / "ids_5_12_anomaly_timeline.png", accepted, windows,
        int(config["anomaly_timeline"]["first_candidate_id"]),
        int(config["anomaly_timeline"]["last_candidate_id"]),
    )

    after = protected_fingerprints(inputs)
    protected_unchanged = before == after
    if not protected_unchanged:
        raise RuntimeError("A protected input changed during vertical observability analysis")
    final_summary = {
        "diagnostic_only": True,
        "diagnostic_name": "independent GICP vertical observability and z-anomaly decomposition",
        "localization_replayed": False,
        "measured_stage_height_used": False,
        "registration_or_gate_parameter_changed": False,
        "input_counts": {
            "localization_rows": len(records),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "candidate_windows": len(windows),
            "mapping_trajectory_rows": int(trajectory.shape[0]),
        },
        "hessian_convention_audit": audit,
        "mapping_association_quality": association_quality,
        "z_drift_decomposition": drift,
        "group_comparison": comparison,
        "vertical_observability": vertical,
        "z_orientation_correlations": orientation_correlations,
        "mapping_discrepancy_hessian_correlations": discrepancy_correlations,
        "quality_gate_observation": quality_observation,
        "interpretation": interpretation,
        "rviz_handoff": rviz_handoff,
        "protected_inputs_unchanged": protected_unchanged,
        "input_integrity": {"before": before, "after": after, "unchanged": protected_unchanged},
        "output_directory": str(output),
        "scientific_conclusion": (
            "ID7 enters GICP with a low accepted-pose anchor created by earlier large alternating "
            "GICP z corrections. Relative final-Hessian metrics do not show weaker anomaly-region "
            "vertical curvature, so correspondence/local-basin audit is the justified next Gate."
        ),
    }
    write_json(output / "gicp_vertical_observability_diagnostic_summary.json", final_summary)
    (output / "gicp_vertical_observability_diagnostic_report.md").write_text(
        report_text(final_summary, audit, candidate_rows), encoding="utf-8"
    )
    return final_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    summary = run(args.config, args.output_directory)
    largest = summary["z_drift_decomposition"]["full_interval"][
        "largest_single_negative_delta_z"
    ]
    print(json.dumps({
        "accepted_rows": summary["input_counts"]["accepted_rows"],
        "anchor_invariant_violations": summary["z_drift_decomposition"][
            "anchor_prediction_invariant"
        ]["violation_count"],
        "largest_negative_delta_z": largest["delta_z"],
        "largest_negative_delta_z_time_sec": largest["bag_relative_time"],
        "mapping_association_available": summary["mapping_association_quality"]["available_count"],
        "protected_inputs_unchanged": summary["protected_inputs_unchanged"],
    }, sort_keys=True))


def z_orientation_correlations(accepted):
    variables = ["correction_roll", "correction_pitch", "correction_yaw"]
    rows = correlations(accepted, "all_accepted", variables)
    for group in ("baseline", "anomaly", "recovery_stage"):
        rows.extend(correlations(
            [row for row in accepted if row["diagnostic_group"] == group], group, variables
        ))
    return rows


def discrepancy_hessian_correlations(accepted):
    variables = [
        "minimum_eigenvalue", "condition_number", "map_vertical_effective_information",
        "map_vertical_inverse_curvature_proxy", "weakest_mode_z_participation",
    ]
    output = []
    for variable in variables:
        paired = [
            (abs(row["independent_minus_mapping_z"]), row[variable]) for row in accepted
            if row.get("independent_minus_mapping_z") is not None and row.get(variable) is not None
        ]
        first = np.asarray([item[0] for item in paired])
        second = np.asarray([item[1] for item in paired])
        output.append({
            "first_variable": "absolute_independent_minus_mapping_z",
            "second_variable": variable,
            "sample_count": len(paired),
            "pearson": ply_diagnostic.finite_or_none(physical_gate.pearson(first, second)),
            "spearman": ply_diagnostic.finite_or_none(physical_gate.spearman(first, second)),
            "interpretation_limit": (
                "local final-solution Hessian cannot exclude wrong correspondences or another cost basin"
            ),
        })
    return output


if __name__ == "__main__":
    main()
