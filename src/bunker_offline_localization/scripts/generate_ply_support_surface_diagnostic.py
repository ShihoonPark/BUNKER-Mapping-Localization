#!/usr/bin/env python3

"""Blind PLY support-surface geometry and LiDAR pose-z cross-check diagnostic."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_ply_support_surface_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_imu_gicp_physical_gate as physical_gate
import generate_surface_pose_z_diagnostic as pose_z_diagnostic

np = physical_gate.np
plt = physical_gate.plt
yaml = physical_gate.yaml

GROUND = "ground_before"
STAGE = "stage_top"
LABELS = (GROUND, STAGE)
SUCCESS = "SUCCESS"
AMBIGUOUS = "AMBIGUOUS_SURFACE"


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


def finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


def load_ply_points(path):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        raise ValueError(f"PLY map has no finite vertices: {path}")
    return points


def estimate_pca_normals(points, knn):
    """Deterministic KNN PCA normals; their sign is intentionally not oriented."""
    import open3d as o3d

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < knn:
        raise ValueError("PCA normal estimation requires finite Nx3 points and enough neighbors")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(cloud)
    normals = np.empty_like(points)
    for index, point in enumerate(points):
        count, neighbors, _ = tree.search_knn_vector_3d(point, int(knn))
        if count < 3:
            raise ValueError(f"PLY normal neighborhood {index} has only {count} points")
        local = points[np.asarray(neighbors, dtype=int)]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / float(count)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        normals[index] = normal / np.linalg.norm(normal)
    return normals


def convex_hull_2d(points):
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float).tolist())))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def point_in_convex_hull(point, hull, tolerance=1.0e-10):
    point = np.asarray(point, dtype=float)
    hull = np.asarray(hull, dtype=float)
    if hull.shape[0] < 3:
        return False
    crosses = np.asarray([
        np.cross(hull[(index + 1) % hull.shape[0]] - hull[index], point - hull[index])
        for index in range(hull.shape[0])
    ])
    return bool(np.all(crosses >= -tolerance) or np.all(crosses <= tolerance))


def connected_components(points, normals, indices, config):
    indices = np.asarray(indices, dtype=int)
    count = indices.size
    parents = np.arange(count, dtype=int)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    xy_link = float(config["component_xy_link_m"])
    z_link = float(config["component_z_link_m"])
    normal_cosine = math.cos(math.radians(float(config["component_normal_angle_deg"])))
    for first in range(count):
        remainder = indices[first + 1:]
        if remainder.size == 0:
            continue
        point = points[indices[first]]
        xy_distance = np.linalg.norm(points[remainder, :2] - point[:2], axis=1)
        z_distance = np.abs(points[remainder, 2] - point[2])
        normal_similarity = np.abs(normals[remainder] @ normals[indices[first]])
        links = np.flatnonzero(
            (xy_distance <= xy_link) & (z_distance <= z_link)
            & (normal_similarity >= normal_cosine)
        )
        for relative in links:
            union(first, first + 1 + int(relative))
    components = {}
    for local_index, global_index in enumerate(indices):
        components.setdefault(find(local_index), []).append(int(global_index))
    return [np.asarray(value, dtype=int) for _, value in sorted(components.items())]


def component_metrics(points, indices, candidate_xy):
    local = points[indices]
    xy = local[:, :2]
    covariance = np.cov(xy, rowvar=False) if xy.shape[0] > 1 else np.zeros((2, 2))
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    hull = convex_hull_2d(xy)
    return {
        "indices": np.asarray(indices, dtype=int),
        "point_count": int(indices.size),
        "center_inside_xy_hull": point_in_convex_hull(candidate_xy, hull),
        "center_min_xy_distance_m": float(np.min(np.linalg.norm(xy - candidate_xy, axis=1))),
        "xy_extent_x_m": float(np.ptp(xy[:, 0])),
        "xy_extent_y_m": float(np.ptp(xy[:, 1])),
        "xy_covariance_eigenvalue_min_m2": float(covariance_eigenvalues[0]),
        "xy_covariance_eigenvalue_max_m2": float(covariance_eigenvalues[-1]),
        "median_z_m": float(np.median(local[:, 2])),
        "minimum_point_index": int(np.min(indices)),
    }


def fit_local_plane(points, indices, candidate_xy, fit_config):
    local = points[indices]
    design = np.column_stack((local[:, 0], local[:, 1], np.ones(local.shape[0])))
    fit = pose_z_diagnostic.huber_irls(design, local[:, 2], fit_config)
    a, b, c = map(float, fit["coefficients"])
    surface_z = a * float(candidate_xy[0]) + b * float(candidate_xy[1]) + c
    return {
        "surface_z_at_candidate_m": float(surface_z),
        "surface_a_dz_dx": a,
        "surface_b_dz_dy": b,
        "surface_c_m": c,
        "surface_slope_deg": math.degrees(math.atan(math.hypot(a, b))),
        "surface_normal_x": float(-a / math.sqrt(a * a + b * b + 1.0)),
        "surface_normal_y": float(-b / math.sqrt(a * a + b * b + 1.0)),
        "surface_normal_z": float(1.0 / math.sqrt(a * a + b * b + 1.0)),
        "surface_design_rank": fit["rank"],
        "surface_condition_number": fit["condition_number"],
        "surface_fit_converged": fit["converged"],
        "surface_fit_iterations": fit["iterations"],
        "surface_rmse_m": fit["residual_metrics"]["rmse_m"],
        "surface_median_abs_residual_m": fit["residual_metrics"]["median_absolute_residual_m"],
        "surface_p95_abs_residual_m": fit["residual_metrics"]["p95_absolute_residual_m"],
    }


def extract_local_surface(points, normals, candidate_xy, config, radius_m):
    """Extract a local support component using geometry only; no pose-z/height input exists."""
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    candidate_xy = np.asarray(candidate_xy, dtype=float)
    local_distance = np.linalg.norm(points[:, :2] - candidate_xy, axis=1)
    neighborhood = np.flatnonzero(local_distance <= float(radius_m))
    support_cosine = math.cos(math.radians(float(config["support_max_tilt_deg"])))
    support = neighborhood[np.abs(normals[neighborhood, 2]) >= support_cosine]
    result = {
        "radius_m": float(radius_m),
        "neighborhood_indices": neighborhood,
        "support_indices": support,
        "selected_indices": np.asarray([], dtype=int),
        "component_metrics": [],
        "status": "NO_SUPPORT_LIKE_POINTS",
        "selection_reason": "no points passed the predeclared normal tilt filter",
    }
    if support.size == 0:
        return result
    raw_components = connected_components(points, normals, support, config)
    minimum_points = int(config["minimum_component_points"])
    minimum_eigenvalue = float(config["minimum_component_xy_cov_eigenvalue_m2"])
    components = []
    for indices in raw_components:
        metrics = component_metrics(points, indices, candidate_xy)
        metrics["sufficient_point_count"] = metrics["point_count"] >= minimum_points
        metrics["sufficient_2d_spread"] = (
            metrics["xy_covariance_eigenvalue_min_m2"] >= minimum_eigenvalue
        )
        if metrics["sufficient_point_count"] and metrics["sufficient_2d_spread"]:
            components.append(metrics)
    result["component_metrics"] = components
    if not components:
        result["status"] = "NO_VALID_COMPONENT"
        result["selection_reason"] = "components failed point-count or 2D-spread requirements"
        return result

    containing = [item for item in components if item["center_inside_xy_hull"]]
    pool = containing if containing else components
    pool = sorted(pool, key=lambda item: (
        item["center_min_xy_distance_m"], -item["point_count"],
        -item["xy_covariance_eigenvalue_min_m2"], item["minimum_point_index"],
    ))
    maximum_distance = float(config["maximum_component_center_distance_m"])
    if not containing and pool[0]["center_min_xy_distance_m"] > maximum_distance:
        result["status"] = "NO_COMPONENT_SUPPORTS_CENTER"
        result["selection_reason"] = "nearest valid component exceeds the predeclared XY distance"
        return result
    ambiguity_margin = float(config["ambiguous_center_distance_margin_m"])
    if len(pool) > 1 and (
        pool[1]["center_min_xy_distance_m"] - pool[0]["center_min_xy_distance_m"]
        <= ambiguity_margin
    ):
        result["status"] = AMBIGUOUS
        result["selection_reason"] = (
            "multiple geometry-only components have indistinguishable candidate-center support"
        )
        result["ambiguous_component_indices"] = [
            item["indices"] for item in pool
            if item["center_min_xy_distance_m"] - pool[0]["center_min_xy_distance_m"]
            <= ambiguity_margin
        ]
        return result

    selected = pool[0]
    try:
        plane = fit_local_plane(points, selected["indices"], candidate_xy, config["robust_fit"])
    except ValueError as error:
        result["status"] = "POOR_PLANE_FIT"
        result["selection_reason"] = str(error)
        return result
    result.update(plane)
    result.update({
        "status": SUCCESS,
        "selection_reason": (
            "candidate center hull support then XY distance, point count, and 2D spread"
        ),
        "selected_indices": selected["indices"],
        "selected_component": selected,
    })
    return result


def load_candidates(path, label_config):
    ground_ids = [int(value) for value in label_config[GROUND]]
    stage_ids = [int(value) for value in label_config[STAGE]]
    if ground_ids != list(range(1, 12)) or stage_ids != list(range(12, 23)):
        raise ValueError("Fixed labels must remain IDs 1-11 ground and IDs 12-22 stage")
    if str(label_config["ground_after"]).lower() != "unavailable":
        raise ValueError("ground_after must remain unavailable")
    rows = read_csv(path)
    if [int(row["diagnostic_id"]) for row in rows] != list(range(1, 23)):
        raise ValueError("Expected frozen diagnostic IDs 1-22 in route order")
    candidates = []
    for row in rows:
        candidate_id = int(row["diagnostic_id"])
        expected = GROUND if candidate_id <= 11 else STAGE
        if row["physical_label"] != expected:
            raise ValueError(f"Candidate {candidate_id} fixed label changed")
        candidates.append({
            "diagnostic_id": candidate_id,
            "physical_label": expected,
            "start_time_sec": float(row["start_time_sec"]),
            "end_time_sec": float(row["end_time_sec"]),
            "sample_count": int(row["sample_count"]),
            "candidate_x_m": float(row["median_x_m"]),
            "candidate_y_m": float(row["median_y_m"]),
            "independent_pose_z_median_m": float(row["median_z_m"]),
        })
    return candidates


def load_localization(path, origin_timestamp, start_sec, end_sec):
    rows = read_csv(path)
    records = []
    for row_index, row in enumerate(rows):
        timestamp = float(row["timestamp"])
        offset = timestamp - float(origin_timestamp)
        if offset < start_sec - 1.0e-9 or offset > end_sec + 1.0e-9:
            raise ValueError(f"Localization row {row_index} is outside the fixed 0-50 s interval")
        records.append({
            "row_index": row_index,
            "timestamp": timestamp,
            "time_sec": offset,
            "accepted": row["accepted"] == "1",
            "x_m": float(row["gicp_x"]),
            "y_m": float(row["gicp_y"]),
            "z_m": float(row["gicp_z"]),
        })
    return records


def load_glim_trajectory(path):
    trajectory = np.loadtxt(path, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.shape[1] != 8 or not np.all(np.isfinite(trajectory)):
        raise ValueError("GLIM trajectory must contain finite timestamp x y z qx qy qz qw rows")
    quaternion_norms = np.linalg.norm(trajectory[:, 4:8], axis=1)
    if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-3:
        raise ValueError("GLIM trajectory contains non-normalized quaternions")
    return trajectory


def monotonic_xy_association(candidate_xy, trajectory_xy):
    """Minimum total XY squared-distance association with nondecreasing trajectory indices."""
    candidate_xy = np.asarray(candidate_xy, dtype=float)
    trajectory_xy = np.asarray(trajectory_xy, dtype=float)
    if (candidate_xy.ndim != 2 or candidate_xy.shape[1] != 2
            or trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2
            or candidate_xy.shape[0] == 0 or trajectory_xy.shape[0] == 0):
        raise ValueError("Monotonic XY association requires non-empty Nx2 and Mx2 inputs")
    distances = np.linalg.norm(
        candidate_xy[:, None, :] - trajectory_xy[None, :, :], axis=2
    )
    candidate_count, trajectory_count = distances.shape
    cost = distances[0] ** 2
    predecessors = np.full((candidate_count, trajectory_count), -1, dtype=int)
    for candidate_index in range(1, candidate_count):
        prefix_cost = np.empty(trajectory_count, dtype=float)
        prefix_index = np.empty(trajectory_count, dtype=int)
        best_cost = math.inf
        best_index = -1
        for trajectory_index in range(trajectory_count):
            # Strict less-than keeps the earliest trajectory index on deterministic ties.
            if cost[trajectory_index] < best_cost:
                best_cost = float(cost[trajectory_index])
                best_index = trajectory_index
            prefix_cost[trajectory_index] = best_cost
            prefix_index[trajectory_index] = best_index
        cost = prefix_cost + distances[candidate_index] ** 2
        predecessors[candidate_index] = prefix_index
    matched = np.empty(candidate_count, dtype=int)
    matched[-1] = int(np.argmin(cost))
    for candidate_index in range(candidate_count - 1, 0, -1):
        matched[candidate_index - 1] = predecessors[candidate_index, matched[candidate_index]]
    return matched, distances[np.arange(candidate_count), matched]


def distribution_metrics(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "median": None, "minimum": None, "maximum": None,
            "range": None, "mad": None, "p05": None, "p95": None,
        }
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "median": median,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
        "mad": float(np.median(np.abs(values - median))),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def trend_metrics(rows, field):
    selected = [row for row in rows if row.get(field) is not None and np.isfinite(row[field])]
    if len(selected) < 2:
        return {"count": len(selected), "linear_slope_per_candidate": None,
                "pearson": None, "spearman": None}
    candidate_ids = np.asarray([row["diagnostic_id"] for row in selected], dtype=float)
    values = np.asarray([row[field] for row in selected], dtype=float)
    slope = np.polyfit(candidate_ids, values, 1)[0]
    return {
        "count": len(selected),
        "linear_slope_per_candidate": float(slope),
        "pearson": finite_or_none(physical_gate.pearson(candidate_ids, values)),
        "spearman": finite_or_none(physical_gate.spearman(candidate_ids, values)),
    }


def crosscheck_values(surface_z, mapping_z, independent_z):
    surface_ok = surface_z is not None and np.isfinite(surface_z)
    mapping_ok = mapping_z is not None and np.isfinite(mapping_z)
    independent_ok = independent_z is not None and np.isfinite(independent_z)
    return {
        "mapping_clearance_m": float(mapping_z - surface_z) if surface_ok and mapping_ok else None,
        "independent_clearance_m": (
            float(independent_z - surface_z) if surface_ok and independent_ok else None
        ),
        "independent_minus_mapping_pose_z_m": (
            float(independent_z - mapping_z) if mapping_ok and independent_ok else None
        ),
    }


def primary_metric_row(candidate, result):
    selected = result.get("selected_component", {})
    return {
        **candidate,
        "local_ply_points_raw": int(result["neighborhood_indices"].size),
        "local_ply_points_support_like": int(result["support_indices"].size),
        "valid_component_count": len(result["component_metrics"]),
        "selected_component_points": selected.get("point_count", 0),
        "surface_fit_status": result["status"],
        "surface_selection_reason": result["selection_reason"],
        "surface_z_at_candidate_m": result.get("surface_z_at_candidate_m"),
        "surface_a_dz_dx": result.get("surface_a_dz_dx"),
        "surface_b_dz_dy": result.get("surface_b_dz_dy"),
        "surface_c_m": result.get("surface_c_m"),
        "surface_slope_deg": result.get("surface_slope_deg"),
        "surface_normal_x": result.get("surface_normal_x"),
        "surface_normal_y": result.get("surface_normal_y"),
        "surface_normal_z": result.get("surface_normal_z"),
        "surface_rmse_m": result.get("surface_rmse_m"),
        "surface_median_abs_residual_m": result.get("surface_median_abs_residual_m"),
        "surface_p95_abs_residual_m": result.get("surface_p95_abs_residual_m"),
        "surface_design_rank": result.get("surface_design_rank"),
        "surface_condition_number": result.get("surface_condition_number"),
        "xy_extent_x_m": selected.get("xy_extent_x_m"),
        "xy_extent_y_m": selected.get("xy_extent_y_m"),
        "xy_covariance_eigenvalue_min_m2": selected.get("xy_covariance_eigenvalue_min_m2"),
        "xy_covariance_eigenvalue_max_m2": selected.get("xy_covariance_eigenvalue_max_m2"),
        "selected_component_center_xy_distance_m": selected.get("center_min_xy_distance_m"),
        "selected_component_center_inside_xy_hull": selected.get("center_inside_xy_hull"),
    }


def association_rows(candidates, trajectory, maximum_distance):
    candidate_xy = np.asarray([
        [item["candidate_x_m"], item["candidate_y_m"]] for item in candidates
    ])
    matched, distances = monotonic_xy_association(candidate_xy, trajectory[:, 1:3])
    rows = []
    for candidate, trajectory_index, distance in zip(candidates, matched, distances):
        available = float(distance) <= float(maximum_distance)
        trajectory_row = trajectory[int(trajectory_index)]
        rows.append({
            "diagnostic_id": candidate["diagnostic_id"],
            "matched_glim_index": int(trajectory_index),
            "matched_glim_timestamp": float(trajectory_row[0]) if available else None,
            "matched_glim_x_m": float(trajectory_row[1]) if available else None,
            "matched_glim_y_m": float(trajectory_row[2]) if available else None,
            "matched_glim_z_m": float(trajectory_row[3]) if available else None,
            "xy_association_distance_m": float(distance),
            "association_status": "AVAILABLE" if available else "UNAVAILABLE_DISTANCE",
            "association_inputs": "ordered_candidate_xy_and_glim_xy_only",
        })
    return rows


def build_crosscheck_rows(primary_rows, associations):
    association_by_id = {row["diagnostic_id"]: row for row in associations}
    rows = []
    for surface in primary_rows:
        association = association_by_id[surface["diagnostic_id"]]
        values = crosscheck_values(
            surface["surface_z_at_candidate_m"], association["matched_glim_z_m"],
            surface["independent_pose_z_median_m"],
        )
        rows.append({
            "diagnostic_id": surface["diagnostic_id"],
            "physical_label": surface["physical_label"],
            "candidate_x_m": surface["candidate_x_m"],
            "candidate_y_m": surface["candidate_y_m"],
            "surface_fit_status": surface["surface_fit_status"],
            "surface_rmse_m": surface["surface_rmse_m"],
            "surface_p95_abs_residual_m": surface["surface_p95_abs_residual_m"],
            "surface_condition_number": surface["surface_condition_number"],
            "ply_local_support_surface_z_m": surface["surface_z_at_candidate_m"],
            "glim_mapping_lidar_z_m": association["matched_glim_z_m"],
            "independent_gicp_lidar_z_m": surface["independent_pose_z_median_m"],
            "mapping_clearance_m": values["mapping_clearance_m"],
            "independent_clearance_m": values["independent_clearance_m"],
            "independent_minus_mapping_pose_z_m": values["independent_minus_mapping_pose_z_m"],
            "glim_xy_association_distance_m": association["xy_association_distance_m"],
            "glim_association_status": association["association_status"],
        })
    return rows


def project_points_to_polyline(points_xy, path_xy):
    points_xy = np.asarray(points_xy, dtype=float)
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.shape[0] < 2:
        raise ValueError("Ramp corridor requires at least two accepted path points")
    segments = path_xy[1:] - path_xy[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1.0e-9
    starts = path_xy[:-1][keep]
    segments = segments[keep]
    lengths = lengths[keep]
    if lengths.size == 0:
        raise ValueError("Ramp corridor path has zero length")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = np.full(points_xy.shape[0], math.inf)
    best_s = np.zeros(points_xy.shape[0])
    for index, (start, segment, length) in enumerate(zip(starts, segments, lengths)):
        fraction = np.clip(((points_xy - start) @ segment) / (length * length), 0.0, 1.0)
        projected = start + fraction[:, None] * segment
        distance = np.linalg.norm(points_xy - projected, axis=1)
        improve = distance < best_distance
        best_distance[improve] = distance[improve]
        best_s[improve] = cumulative[index] + fraction[improve] * length
    return best_s, best_distance, float(np.sum(lengths))


def ramp_corridor_analysis(points, normals, records, candidates, trajectory, config):
    by_id = {item["diagnostic_id"]: item for item in candidates}
    lower_id = int(config["lower_candidate_id"])
    upper_id = int(config["upper_candidate_id"])
    start = by_id[lower_id]["end_time_sec"]
    end = by_id[upper_id]["start_time_sec"]
    path_records = [
        record for record in records
        if record["accepted"] and start <= record["time_sec"] <= end
    ]
    if len(path_records) < 2:
        return [], [], {
            "status": "RAMP_PROFILE_NOT_RELIABLY_EXTRACTABLE",
            "reason": "fewer than two accepted poses in fixed candidate 11-to-12 interval",
            "time_window_sec": [start, end],
        }
    path_xy = np.asarray([[row["x_m"], row["y_m"]] for row in path_records])
    path_s, _, path_length = project_points_to_polyline(path_xy, path_xy)
    map_s, lateral_distance, _ = project_points_to_polyline(points[:, :2], path_xy)
    tilt_deg = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))
    selected = (
        (lateral_distance <= float(config["corridor_half_width_m"]))
        & (tilt_deg <= float(config["support_max_tilt_deg"]))
    )
    point_rows = []
    for index in np.flatnonzero(selected):
        point_rows.append({
            "map_point_index": int(index),
            "path_arc_length_s_m": float(map_s[index]),
            "lateral_distance_m": float(lateral_distance[index]),
            "ply_x_m": float(points[index, 0]),
            "ply_y_m": float(points[index, 1]),
            "ply_z_m": float(points[index, 2]),
            "normal_abs_nz": float(abs(normals[index, 2])),
            "normal_tilt_deg": float(tilt_deg[index]),
        })
    mapping_indices, mapping_distances = monotonic_xy_association(path_xy, trajectory[:, 1:3])
    path_rows = []
    for record, arc_length, mapping_index, mapping_distance in zip(
            path_records, path_s, mapping_indices, mapping_distances):
        mapping_available = mapping_distance <= 0.60
        path_rows.append({
            "time_sec": record["time_sec"],
            "path_arc_length_s_m": float(arc_length),
            "independent_x_m": record["x_m"],
            "independent_y_m": record["y_m"],
            "independent_z_m": record["z_m"],
            "matched_glim_index": int(mapping_index),
            "glim_xy_distance_m": float(mapping_distance),
            "glim_z_m": float(trajectory[mapping_index, 3]) if mapping_available else None,
        })
    point_z = [row["ply_z_m"] for row in point_rows]
    metrics = {
        "status": "RAW_SUPPORT_LIKE_SCATTER_ONLY" if point_rows else "RAMP_PROFILE_NOT_RELIABLY_EXTRACTABLE",
        "reason": (
            "multi-surface PLY points are retained; no height- or pose-z-selected single curve"
            if point_rows else "no PLY points passed the predeclared corridor and normal filters"
        ),
        "time_window_sec": [start, end],
        "fixed_topology": f"candidate_{lower_id}_end_to_candidate_{upper_id}_start",
        "accepted_path_pose_count": len(path_rows),
        "path_length_m": path_length,
        "support_like_ply_point_count": len(point_rows),
        "support_like_ply_z_metrics": distribution_metrics(point_z),
        "corridor_half_width_m": float(config["corridor_half_width_m"]),
        "support_max_tilt_deg": float(config["support_max_tilt_deg"]),
        "single_ramp_curve_claimed": False,
    }
    return point_rows, path_rows, metrics


def fixed_landing_separation(rows, config):
    by_id = {row["diagnostic_id"]: row for row in rows}
    lower_ids = [int(value) for value in config["lower_landing_candidate_ids"]]
    upper_ids = [int(value) for value in config["upper_landing_candidate_ids"]]

    def landing(candidate_ids):
        entries = [by_id[candidate_id] for candidate_id in candidate_ids]
        valid = [
            row["ply_local_support_surface_z_m"] for row in entries
            if row["surface_fit_status"] == SUCCESS
            and row["ply_local_support_surface_z_m"] is not None
        ]
        return {
            "fixed_candidate_ids": candidate_ids,
            "candidate_statuses": {
                str(row["diagnostic_id"]): row["surface_fit_status"] for row in entries
            },
            "valid_candidate_ids": [
                row["diagnostic_id"] for row in entries if row["surface_fit_status"] == SUCCESS
            ],
            "failed_or_ambiguous_candidate_ids": [
                row["diagnostic_id"] for row in entries if row["surface_fit_status"] != SUCCESS
            ],
            "surface_z_values_m": valid,
            "robust_median_surface_z_m": float(np.median(valid)) if valid else None,
        }

    lower = landing(lower_ids)
    upper = landing(upper_ids)
    available = (
        lower["robust_median_surface_z_m"] is not None
        and upper["robust_median_surface_z_m"] is not None
    )
    return {
        "blind": True,
        "selection_inputs": "first-ramp_route_topology_only",
        "measured_height_used": False,
        "aggregation": config["aggregation"],
        "lower_landing": lower,
        "upper_landing": upper,
        "status": "AVAILABLE" if available else "UNAVAILABLE_FIXED_SET_EXTRACTION_FAILURE",
        "map_landing_separation_m": (
            upper["robust_median_surface_z_m"] - lower["robust_median_surface_z_m"]
            if available else None
        ),
    }


def group_metrics(crosscheck_rows):
    fields = {
        "surface_z": "ply_local_support_surface_z_m",
        "independent_clearance": "independent_clearance_m",
        "mapping_clearance": "mapping_clearance_m",
        "independent_minus_mapping": "independent_minus_mapping_pose_z_m",
        "mapping_pose_z": "glim_mapping_lidar_z_m",
        "independent_pose_z": "independent_gicp_lidar_z_m",
    }
    output = {}
    for label in LABELS:
        selected = [row for row in crosscheck_rows if row["physical_label"] == label]
        output[label] = {
            name: distribution_metrics([row[field] for row in selected])
            for name, field in fields.items()
        }
        output[label]["surface_residual_quality"] = {
            "rmse_m": distribution_metrics([
                row.get("surface_rmse_m") for row in selected if "surface_rmse_m" in row
            ])
        }
    return output


def progression_metrics(rows):
    fields = [
        "ply_local_support_surface_z_m", "glim_mapping_lidar_z_m",
        "independent_gicp_lidar_z_m", "mapping_clearance_m",
        "independent_clearance_m", "independent_minus_mapping_pose_z_m",
    ]
    scopes = {"all_candidates": rows}
    scopes.update({label: [row for row in rows if row["physical_label"] == label] for label in LABELS})
    return {
        scope: {field: trend_metrics(selected, field) for field in fields}
        for scope, selected in scopes.items()
    }


def surface_failure_counts(rows):
    counts = {}
    for row in rows:
        counts[row["surface_fit_status"]] = counts.get(row["surface_fit_status"], 0) + 1
    return counts


def surface_quality_metrics(rows, label):
    selected = [
        row for row in rows
        if row["physical_label"] == label and row["surface_fit_status"] == SUCCESS
    ]
    return {
        "surface_z": distribution_metrics([row["surface_z_at_candidate_m"] for row in selected]),
        "plane_rmse": distribution_metrics([row["surface_rmse_m"] for row in selected]),
        "plane_p95_abs_residual": distribution_metrics([
            row["surface_p95_abs_residual_m"] for row in selected
        ]),
        "plane_condition_number": distribution_metrics([
            row["surface_condition_number"] for row in selected
        ]),
        "surface_z_progression": trend_metrics(selected, "surface_z_at_candidate_m"),
    }


def sensitivity_rows(candidates, results_by_radius, primary_radius, landing_config):
    primary = results_by_radius[primary_radius]
    rows = []
    for radius in sorted(results_by_radius):
        results = results_by_radius[radius]
        metric_rows = [primary_metric_row(candidate, result) for candidate, result in zip(candidates, results)]
        pseudo_crosscheck = [{
            **row,
            "ply_local_support_surface_z_m": row["surface_z_at_candidate_m"],
        } for row in metric_rows]
        landing = fixed_landing_separation(pseudo_crosscheck, landing_config)
        for index, (candidate, result) in enumerate(zip(candidates, results)):
            current_z = result.get("surface_z_at_candidate_m")
            primary_z = primary[index].get("surface_z_at_candidate_m")
            current_clearance = (
                candidate["independent_pose_z_median_m"] - current_z
                if current_z is not None else None
            )
            primary_clearance = (
                candidate["independent_pose_z_median_m"] - primary_z
                if primary_z is not None else None
            )
            rows.append({
                "radius_m": radius,
                "is_primary_radius": radius == primary_radius,
                "diagnostic_id": candidate["diagnostic_id"],
                "surface_fit_status": result["status"],
                "surface_z_at_candidate_m": current_z,
                "surface_z_change_vs_primary_m": (
                    current_z - primary_z if current_z is not None and primary_z is not None else None
                ),
                "independent_clearance_m": current_clearance,
                "independent_clearance_change_vs_primary_m": (
                    current_clearance - primary_clearance
                    if current_clearance is not None and primary_clearance is not None else None
                ),
                "radius_surface_success_count": sum(item["status"] == SUCCESS for item in results),
                "radius_ambiguous_count": sum(item["status"] == AMBIGUOUS for item in results),
                "radius_failure_count": sum(
                    item["status"] not in (SUCCESS, AMBIGUOUS) for item in results
                ),
                "radius_map_landing_separation_m": landing["map_landing_separation_m"],
            })
    return rows


def plot_map_extraction(path, points, accepted_records, candidates, results, radius, maximum_points):
    plot_points = points
    if maximum_points > 0 and points.shape[0] > maximum_points:
        indices = np.linspace(0, points.shape[0] - 1, maximum_points, dtype=int)
        plot_points = points[indices]
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.scatter(plot_points[:, 0], plot_points[:, 1], s=1, color="0.75", alpha=0.35, label="PLY XY")
    accepted = [record for record in accepted_records if record["accepted"]]
    axis.plot([row["x_m"] for row in accepted], [row["y_m"] for row in accepted],
              color="black", linewidth=1.1, label="accepted independent GICP")
    styles = {
        SUCCESS: ("tab:green", "o", "success"),
        AMBIGUOUS: ("tab:orange", "D", "ambiguous"),
    }
    seen = set()
    for candidate, result in zip(candidates, results):
        color, marker, label = styles.get(result["status"], ("tab:red", "x", "failed"))
        legend_label = label if label not in seen else None
        seen.add(label)
        x_value, y_value = candidate["candidate_x_m"], candidate["candidate_y_m"]
        axis.scatter([x_value], [y_value], s=55, color=color, marker=marker,
                     edgecolors="black" if marker != "x" else None, linewidths=0.6,
                     zorder=4, label=legend_label)
        axis.add_patch(plt.Circle((x_value, y_value), radius, fill=False, color=color,
                                  linewidth=0.55, alpha=0.4))
        axis.annotate(str(candidate["diagnostic_id"]), (x_value, y_value), xytext=(4, 4),
                      textcoords="offset points", fontsize=9, weight="bold", color=color)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.set_title(f"Blind local PLY support extraction (primary radius {radius:.2f} m)")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_candidate_montage(path, points, candidates, results, radius, columns):
    rows = int(math.ceil(len(candidates) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.5 * rows),
                                constrained_layout=True, squeeze=False)
    for axis, candidate, result in zip(axes.flat, candidates, results):
        neighborhood = result["neighborhood_indices"]
        support = result["support_indices"]
        selected = result["selected_indices"]
        if neighborhood.size:
            axis.scatter(points[neighborhood, 0], points[neighborhood, 1], s=14,
                         color="0.75", label="all local PLY")
        if support.size:
            axis.scatter(points[support, 0], points[support, 1], s=16,
                         color="tab:orange", label="support-like")
        if result["status"] == AMBIGUOUS:
            for component in result.get("ambiguous_component_indices", []):
                axis.scatter(points[component, 0], points[component, 1], s=24,
                             facecolors="none", edgecolors="tab:purple")
        if selected.size:
            axis.scatter(points[selected, 0], points[selected, 1], s=24,
                         color="tab:green", label="selected")
        center = (candidate["candidate_x_m"], candidate["candidate_y_m"])
        axis.scatter([center[0]], [center[1]], marker="+", s=100, linewidths=2,
                     color="black", label="candidate center")
        axis.add_patch(plt.Circle(center, radius, fill=False, color="0.4", linewidth=0.7))
        axis.set_title(f"ID {candidate['diagnostic_id']}: {result['status']}", fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(candidates):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.suptitle(
        "Geometry-only component selection: neither pose-z nor measured height selects a surface",
        fontsize=13,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_three_way(path, rows):
    ids = np.asarray([row["diagnostic_id"] for row in rows])
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for field, label, style in (
        ("ply_local_support_surface_z_m", "PLY local support surface z", "o-"),
        ("glim_mapping_lidar_z_m", "canonical GLIM mapping LiDAR z", "s-"),
        ("independent_gicp_lidar_z_m", "independent GICP LiDAR z", "^-"),
    ):
        values = np.asarray([
            np.nan if row[field] is None else row[field] for row in rows
        ], dtype=float)
        axis.plot(ids, values, style, linewidth=1.4, markersize=5, label=label)
    axis.axvline(11.5, color="0.3", linestyle="--", linewidth=1, label="fixed label boundary")
    axis.set_xticks(ids)
    axis.set_xlabel("diagnostic candidate ID (route order)")
    axis.set_ylabel("map-frame z [m]")
    axis.set_title("Three-way z comparison (missing PLY surfaces are not imputed)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_clearance(path, rows):
    ids = [row["diagnostic_id"] for row in rows]
    mapping = [np.nan if row["mapping_clearance_m"] is None else row["mapping_clearance_m"] for row in rows]
    independent = [np.nan if row["independent_clearance_m"] is None else row["independent_clearance_m"] for row in rows]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(ids, mapping, "s-", label="mapping LiDAR - PLY surface")
    axis.plot(ids, independent, "o-", label="independent LiDAR - PLY surface")
    axis.axvline(11.5, color="0.3", linestyle="--", linewidth=1)
    axis.set_xticks(ids)
    axis.set_xlabel("diagnostic candidate ID")
    axis.set_ylabel("LiDAR clearance [m] (absolute expected value unknown)")
    axis.set_title("LiDAR-to-local-PLY clearance comparison")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_independent_minus_mapping(path, rows):
    ids = [row["diagnostic_id"] for row in rows]
    values = [
        np.nan if row["independent_minus_mapping_pose_z_m"] is None
        else row["independent_minus_mapping_pose_z_m"] for row in rows
    ]
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.plot(ids, values, "o-", color="tab:purple")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(11.5, color="0.3", linestyle="--", linewidth=1)
    axis.set_xticks(ids)
    axis.set_xlabel("diagnostic candidate ID")
    axis.set_ylabel("independent z - mapping z [m]")
    axis.set_title("Independent minus canonical mapping LiDAR pose-z")
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ramp_corridor(path, point_rows, path_rows):
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    if point_rows:
        scatter = axis.scatter(
            [row["path_arc_length_s_m"] for row in point_rows],
            [row["ply_z_m"] for row in point_rows],
            c=[row["lateral_distance_m"] for row in point_rows], cmap="viridis_r",
            s=11, alpha=0.6, label="raw nearby support-like PLY points",
        )
        figure.colorbar(scatter, ax=axis, label="XY lateral distance to path [m]")
    if path_rows:
        axis.plot([row["path_arc_length_s_m"] for row in path_rows],
                  [row["independent_z_m"] for row in path_rows], color="tab:red",
                  linewidth=1.5, label="independent LiDAR z")
        available = [row for row in path_rows if row["glim_z_m"] is not None]
        axis.plot([row["path_arc_length_s_m"] for row in available],
                  [row["glim_z_m"] for row in available], color="tab:blue",
                  linewidth=1.2, label="XY-associated mapping LiDAR z")
    axis.set_xlabel("accepted independent path arc length s [m]")
    axis.set_ylabel("map-frame z [m]")
    axis.set_title("First-ramp corridor: raw PLY s-z scatter (no automatic surface curve)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_posthoc_height(path, posthoc):
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if posthoc["blind_map_landing_separation_m"] is not None:
        axis.bar([0], [posthoc["blind_map_landing_separation_m"]], width=0.5,
                 color="tab:blue", label="blind PLY landing separation")
    axis.axhline(posthoc["measured_stage_height_m"], color="tab:red", linestyle="--",
                 label="manual measured 0.150 m")
    axis.set_xticks([0], ["fixed IDs 10-11 vs 12-13"])
    axis.set_ylabel("signed upper-minus-lower height [m]")
    axis.set_title("Phase B post-hoc only: measured height did not select PLY surfaces")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def blind_input_fingerprints(inputs):
    """Fingerprint only height-blind inputs; physical config is deliberately absent."""
    return {
        "bag": physical_gate.fingerprint(inputs["bag"]),
        "map_ply": physical_gate.fingerprint(inputs["map_ply"], include_hash=True),
        "glim_mapping_trajectory": physical_gate.fingerprint(
            inputs["glim_mapping_trajectory"], include_hash=True
        ),
        "localization_csv": physical_gate.fingerprint(inputs["localization_csv"], include_hash=True),
        "plateau_diagnostic_directory": physical_gate.fingerprint(
            inputs["plateau_diagnostic_directory"]
        ),
        "pose_z_diagnostic_directory": physical_gate.fingerprint(
            inputs["pose_z_diagnostic_directory"]
        ),
    }


def posthoc_height(blind_landing, physical_config_path):
    with Path(physical_config_path).open(encoding="utf-8") as stream:
        physical_config = yaml.safe_load(stream)["physical_stage_height"]
    if not physical_config.get("available", False):
        raise ValueError("The manual physical stage height must remain available for Phase B")
    measured = float(physical_config["height_m"])
    tolerance = float(physical_config["tolerance_m"])
    estimate = blind_landing["map_landing_separation_m"]
    if estimate is None:
        return {
            "phase": "B_posthoc_after_blind_sha256_seal",
            "measured_stage_height_m": measured,
            "blind_map_landing_separation_m": None,
            "signed_error_m": None,
            "absolute_error_m": None,
            "relative_error_percent": None,
            "initial_screening_tolerance_m": tolerance,
            "tolerance_interpretation": "initial screening threshold, not calibrated accuracy",
            "status": "MAP_SURFACE_HEIGHT_SCREEN_WARN_UNAVAILABLE",
            "production_localization_failure": False,
        }
    signed_error = float(estimate - measured)
    absolute_error = abs(signed_error)
    return {
        "phase": "B_posthoc_after_blind_sha256_seal",
        "measured_stage_height_m": measured,
        "blind_map_landing_separation_m": estimate,
        "signed_error_m": signed_error,
        "absolute_error_m": absolute_error,
        "relative_error_percent": 100.0 * absolute_error / measured,
        "initial_screening_tolerance_m": tolerance,
        "tolerance_interpretation": "initial screening threshold, not calibrated accuracy",
        "status": (
            "MAP_SURFACE_HEIGHT_SCREEN_PASS"
            if absolute_error <= tolerance else "MAP_SURFACE_HEIGHT_SCREEN_FAIL"
        ),
        "production_localization_failure": False,
    }


def build_blind_summary(config, primary_rows, crosscheck_rows, association_rows_value,
                        sensitivity, ramp_metrics, landing, input_fingerprints):
    failures = surface_failure_counts(primary_rows)
    group = group_metrics(crosscheck_rows)
    association_distances = [
        row["xy_association_distance_m"] for row in association_rows_value
        if row["association_status"] == "AVAILABLE"
    ]
    independent_minus_mapping = distribution_metrics([
        row["independent_minus_mapping_pose_z_m"] for row in crosscheck_rows
    ])
    negative_clearance_ids = [
        row["diagnostic_id"] for row in crosscheck_rows
        if row["mapping_clearance_m"] is not None and row["mapping_clearance_m"] < 0.0
        and row["independent_clearance_m"] is not None and row["independent_clearance_m"] < 0.0
    ]
    radius_summaries = {}
    for radius in sorted(set(row["radius_m"] for row in sensitivity)):
        radius_rows = [row for row in sensitivity if row["radius_m"] == radius]
        radius_summaries[str(radius)] = {
            "surface_success_count": radius_rows[0]["radius_surface_success_count"],
            "ambiguous_count": radius_rows[0]["radius_ambiguous_count"],
            "failure_count": radius_rows[0]["radius_failure_count"],
            "map_landing_separation_m": radius_rows[0]["radius_map_landing_separation_m"],
            "surface_z_change_vs_primary_m": distribution_metrics([
                row["surface_z_change_vs_primary_m"] for row in radius_rows
            ]),
            "clearance_change_vs_primary_m": distribution_metrics([
                row["independent_clearance_change_vs_primary_m"] for row in radius_rows
            ]),
        }
    return {
        "diagnostic_only": True,
        "analysis_phase": "A_blind_before_measured_height_file_read",
        "measured_height_used_for_surface_selection": False,
        "measured_height_used_for_glim_association": False,
        "measured_height_used_for_blind_map_height": False,
        "independent_pose_z_used_for_surface_selection": False,
        "fixed_labels": {
            GROUND: list(range(1, 12)),
            STAGE: list(range(12, 23)),
            "ground_after": "unavailable",
        },
        "surface_extraction_configuration": config["surface_extraction"],
        "primary_candidate_radius_m": float(
            config["surface_extraction"]["primary_candidate_radius_m"]
        ),
        "surface_success_count": failures.get(SUCCESS, 0),
        "surface_ambiguous_count": failures.get(AMBIGUOUS, 0),
        "surface_failure_count": sum(
            count for status, count in failures.items() if status not in (SUCCESS, AMBIGUOUS)
        ),
        "surface_failure_reasons": failures,
        "ground_surface_z_metrics": surface_quality_metrics(primary_rows, GROUND),
        "stage_surface_z_metrics": surface_quality_metrics(primary_rows, STAGE),
        "ground_independent_clearance_metrics": group[GROUND]["independent_clearance"],
        "stage_independent_clearance_metrics": group[STAGE]["independent_clearance"],
        "ground_mapping_clearance_metrics": group[GROUND]["mapping_clearance"],
        "stage_mapping_clearance_metrics": group[STAGE]["mapping_clearance"],
        "independent_minus_mapping_metrics": {
            "all": independent_minus_mapping,
            GROUND: group[GROUND]["independent_minus_mapping"],
            STAGE: group[STAGE]["independent_minus_mapping"],
        },
        "glim_xy_association": {
            "method": config["glim_association"]["method"],
            "inputs": "candidate_xy_and_canonical_glim_xy_only",
            "successful_candidates": sum(
                row["association_status"] == "AVAILABLE" for row in association_rows_value
            ),
            "distance_metrics_m": distribution_metrics(association_distances),
            "maximum_allowed_distance_m": float(
                config["glim_association"]["max_glim_xy_association_distance_m"]
            ),
        },
        "candidate_progression_metrics": progression_metrics(crosscheck_rows),
        "ramp_corridor_metrics": ramp_metrics,
        "blind_map_landing_separation": landing,
        "surface_extraction_sensitivity": radius_summaries,
        "post_extraction_multi_surface_warning": {
            "candidate_ids_with_both_lidar_poses_below_selected_horizontal_layer": negative_clearance_ids,
            "used_to_reselect_or_remove_surface": False,
            "interpretation": (
                "The XY-only rule selected a horizontal layer above both LiDAR poses. This "
                "post-extraction warning demonstrates unresolved multi-surface geometry; it "
                "does not retroactively choose another component."
            ),
        },
        "blind_input_fingerprints": input_fingerprints,
        "historical_height_status_preserved": "HEIGHT_FAIL",
        "production_localization_failure": False,
    }


def determine_interpretation(blind):
    success = blind["surface_success_count"]
    ambiguous_or_failed = blind["surface_ambiguous_count"] + blind["surface_failure_count"]
    multi_surface_ids = blind["post_extraction_multi_surface_warning"][
        "candidate_ids_with_both_lidar_poses_below_selected_horizontal_layer"
    ]
    independent_range = max(
        value for value in (
            blind["ground_independent_clearance_metrics"]["range"],
            blind["stage_independent_clearance_metrics"]["range"],
        ) if value is not None
    )
    mapping_range = max(
        value for value in (
            blind["ground_mapping_clearance_metrics"]["range"],
            blind["stage_mapping_clearance_metrics"]["range"],
        ) if value is not None
    )
    # This is an evidence summary, not a surface-selection rule or calibrated Gate threshold.
    if success < 11:
        label = "insufficient_ply_support"
    elif ambiguous_or_failed > 0 and multi_surface_ids:
        label = "mixed_with_insufficient_local_ply_support"
    elif independent_range > mapping_range:
        label = "independent_localization_contribution_plausible"
    elif mapping_range > 0.0:
        label = "mapping_trajectory_or_map_geometry_contribution_plausible"
    else:
        label = "unresolved"
    return {
        "most_supported_interpretation": label,
        "automatic_causal_claim": False,
        "selection_threshold_used": False,
        "reason": (
            "PLY support geometry, mapping clearance, independent clearance, and extraction "
            "ambiguity are reported jointly; the result does not identify GLIM's internal cause."
        ),
    }


def format_value(value, digits=6):
    return "UNAVAILABLE" if value is None else f"{value:.{digits}f}"


def report_text(summary, crosscheck_rows):
    blind = summary["blind_analysis"]
    ground_surface = blind["ground_surface_z_metrics"]
    stage_surface = blind["stage_surface_z_metrics"]
    association = blind["glim_xy_association"]
    landing = blind["blind_map_landing_separation"]
    posthoc = summary["posthoc_map_height_error"]
    focus_rows = [row for row in crosscheck_rows if 6 <= row["diagnostic_id"] <= 11]
    focus_table = "\n".join(
        f"| {row['diagnostic_id']} | {row['surface_fit_status']} | "
        f"{format_value(row['ply_local_support_surface_z_m'])} | "
        f"{format_value(row['glim_mapping_lidar_z_m'])} | "
        f"{format_value(row['independent_gicp_lidar_z_m'])} | "
        f"{format_value(row['mapping_clearance_m'])} | "
        f"{format_value(row['independent_clearance_m'])} |"
        for row in focus_rows
    )
    stage_rows = [row for row in crosscheck_rows if row["physical_label"] == STAGE]
    stage_table = "\n".join(
        f"| {row['diagnostic_id']} | {row['surface_fit_status']} | "
        f"{format_value(row['ply_local_support_surface_z_m'])} | "
        f"{format_value(row['glim_mapping_lidar_z_m'])} | "
        f"{format_value(row['independent_gicp_lidar_z_m'])} |"
        for row in stage_rows
    )
    sensitivity_lines = "\n".join(
        f"- radius {radius} m: success={metrics['surface_success_count']}, "
        f"ambiguous={metrics['ambiguous_count']}, failure={metrics['failure_count']}, "
        f"landing separation={format_value(metrics['map_landing_separation_m'])} m"
        for radius, metrics in blind["surface_extraction_sensitivity"].items()
    )
    return f"""# PLY local support-surface geometry diagnostic Gate

This is a read-only, diagnostic-only post-processing Gate. It did not replay localization or
change EKF, small_gicp, voxel, registration, map, GLIM, bag, or prior results.

## Blind geometry extraction

- Primary candidate XY radius: {blind['primary_candidate_radius_m']:.2f} m (fixed before analysis)
- Success / ambiguous / other failure: {blind['surface_success_count']} / {blind['surface_ambiguous_count']} / {blind['surface_failure_count']}
- Normal KNN: {blind['surface_extraction_configuration']['normal_knn']}
- Measured height used for surface selection / GLIM association / blind map height: false / false / false
- Independent pose-z used for surface selection: false
- Blind summary SHA256: `{summary['blind_sha256']}`

Sensitivity is descriptive only; it does not replace the 0.60 m primary result.

{sensitivity_lines}

## Ground and stage PLY surface-z

- Ground valid surface-z range: {format_value(ground_surface['surface_z']['range'])} m;
  median={format_value(ground_surface['surface_z']['median'])} m;
  progression Spearman={format_value(ground_surface['surface_z_progression']['spearman'], 4)}.
- Ground plane RMSE median/p95-of-RMSE: {format_value(ground_surface['plane_rmse']['median'])} /
  {format_value(ground_surface['plane_rmse']['p95'])} m.
- Stage valid surface-z range: {format_value(stage_surface['surface_z']['range'])} m;
  median={format_value(stage_surface['surface_z']['median'])} m;
  progression Spearman={format_value(stage_surface['surface_z_progression']['spearman'], 4)}.
- Stage plane RMSE median/p95-of-RMSE: {format_value(stage_surface['plane_rmse']['median'])} /
  {format_value(stage_surface['plane_rmse']['p95'])} m.
- Post-extraction multi-surface warning IDs (selected layer above both LiDAR poses):
  {blind['post_extraction_multi_surface_warning']['candidate_ids_with_both_lidar_poses_below_selected_horizontal_layer']}.

Large surface range must be read together with component ambiguity and multi-surface PLY sparsity;
it is not automatically a floor-height estimate.

## IDs 6-11 direct cross-check

| ID | PLY status | PLY surface z [m] | mapping LiDAR z [m] | independent LiDAR z [m] | mapping clearance [m] | independent clearance [m] |
|---:|---|---:|---:|---:|---:|---:|
{focus_table}

## Stage IDs 12-22

| ID | PLY status | PLY surface z [m] | mapping LiDAR z [m] | independent LiDAR z [m] |
|---:|---|---:|---:|---:|
{stage_table}

## Canonical GLIM XY-only association

- Available: {association['successful_candidates']}/22 candidates
- XY distance median/p95/max: {format_value(association['distance_metrics_m']['median'])} /
  {format_value(association['distance_metrics_m']['p95'])} /
  {format_value(association['distance_metrics_m']['maximum'])} m
- Association uses ordered candidate XY and canonical GLIM XY only. Different-bag timestamps,
  z, quaternion, and measured height are excluded from its objective.

## Clearances and ramp corridor

- Ground mapping clearance range: {format_value(blind['ground_mapping_clearance_metrics']['range'])} m
- Ground independent clearance range: {format_value(blind['ground_independent_clearance_metrics']['range'])} m
- Stage mapping clearance range: {format_value(blind['stage_mapping_clearance_metrics']['range'])} m
- Stage independent clearance range: {format_value(blind['stage_independent_clearance_metrics']['range'])} m
- Independent-minus-mapping z range: {format_value(blind['independent_minus_mapping_metrics']['all']['range'])} m
- Ramp corridor: {blind['ramp_corridor_metrics']['status']}; raw support-like PLY points=
  {blind['ramp_corridor_metrics']['support_like_ply_point_count']}, path length=
  {blind['ramp_corridor_metrics']['path_length_m']:.3f} m. No single ramp surface curve is claimed.

Absolute clearance is not assigned a physical expected value because `T_base_lidar` is unknown.

## Blind landing separation and Phase B post-hoc comparison

- Lower fixed topology IDs: {landing['lower_landing']['fixed_candidate_ids']}; valid IDs:
  {landing['lower_landing']['valid_candidate_ids']}; failures/ambiguities:
  {landing['lower_landing']['failed_or_ambiguous_candidate_ids']}
- Upper fixed topology IDs: {landing['upper_landing']['fixed_candidate_ids']}; valid IDs:
  {landing['upper_landing']['valid_candidate_ids']}; failures/ambiguities:
  {landing['upper_landing']['failed_or_ambiguous_candidate_ids']}
- Blind PLY upper-minus-lower landing separation: {format_value(landing['map_landing_separation_m'])} m
- Measured stage height (read only after blind sealing): {posthoc['measured_stage_height_m']:.3f} m
- Signed / absolute / relative error: {format_value(posthoc['signed_error_m'])} m /
  {format_value(posthoc['absolute_error_m'])} m / {format_value(posthoc['relative_error_percent'], 3)}%
- Post-hoc status: **{posthoc['status']}**

The 0.050 m tolerance is an initial screening threshold, not calibrated accuracy, and this status
is not a production-localization failure. Historical `HEIGHT_FAIL` remains preserved.

## Interpretation and limitations

Most supported: **{summary['interpretation']['most_supported_interpretation']}**. The sparse PLY
contains multiple plausible horizontal components around some candidate XY locations. Surface
failures and ambiguities remain visible; surfaces are never selected by z closeness. These data do
not establish exact physical map-floor tilt, exact `T_base_lidar`, a lever-arm correction, GLIM's
internal causal failure, absolute localization accuracy, or ground-after height.

Protected inputs unchanged: {summary['protected_inputs_unchanged']}.
"""


def run(config_path, output_directory=None):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    inputs = config["inputs"]
    output = Path(output_directory or config["output_directory"]).resolve()
    protected = [
        inputs["bag"], inputs["map_ply"], inputs["glim_mapping_trajectory"],
        inputs["localization_csv"], inputs["plateau_diagnostic_directory"],
        inputs["pose_z_diagnostic_directory"], inputs["physical_gate_config"],
    ]
    physical_gate.validate_output_path(output, protected)
    before_blind = blind_input_fingerprints(inputs)
    output.mkdir(parents=True, exist_ok=True)

    candidate_source = (
        Path(inputs["pose_z_diagnostic_directory"]) / "surface_candidate_metrics.csv"
    )
    candidates = load_candidates(candidate_source, config["fixed_physical_labels"])
    time_config = config["time_window"]
    records = load_localization(
        inputs["localization_csv"], float(time_config["origin_timestamp"]),
        float(time_config["start_sec"]), float(time_config["end_sec"]),
    )
    points = load_ply_points(inputs["map_ply"])
    surface_config = config["surface_extraction"]
    normals = estimate_pca_normals(points, int(surface_config["normal_knn"]))
    trajectory = load_glim_trajectory(inputs["glim_mapping_trajectory"])

    primary_radius = float(surface_config["primary_candidate_radius_m"])
    radii = [primary_radius] + [
        float(value) for value in surface_config["sensitivity_candidate_radii_m"]
    ]
    if len(set(radii)) != len(radii) or primary_radius != 0.60:
        raise ValueError("Primary radius must remain 0.60 m and sensitivity radii must be distinct")
    results_by_radius = {}
    for radius in radii:
        results_by_radius[radius] = [
            extract_local_surface(
                points, normals, [candidate["candidate_x_m"], candidate["candidate_y_m"]],
                surface_config, radius,
            )
            for candidate in candidates
        ]
    primary_results = results_by_radius[primary_radius]
    primary_rows = [
        primary_metric_row(candidate, result)
        for candidate, result in zip(candidates, primary_results)
    ]
    associations = association_rows(
        candidates, trajectory,
        float(config["glim_association"]["max_glim_xy_association_distance_m"]),
    )
    crosscheck = build_crosscheck_rows(primary_rows, associations)
    sensitivity = sensitivity_rows(
        candidates, results_by_radius, primary_radius, config["landing_comparison"]
    )
    ramp_points, ramp_path, ramp_metrics = ramp_corridor_analysis(
        points, normals, records, candidates, trajectory, config["ramp_corridor"]
    )
    landing = fixed_landing_separation(crosscheck, config["landing_comparison"])
    clearance_metrics = group_metrics(crosscheck)

    write_csv(output / "candidate_ply_surface_metrics.csv", primary_rows)
    write_csv(output / "glim_xy_association.csv", associations)
    write_csv(output / "candidate_lidar_surface_crosscheck.csv", crosscheck)
    write_csv(output / "surface_extraction_sensitivity.csv", sensitivity)
    ramp_fields = [
        "map_point_index", "path_arc_length_s_m", "lateral_distance_m", "ply_x_m",
        "ply_y_m", "ply_z_m", "normal_abs_nz", "normal_tilt_deg",
    ]
    write_csv(output / "ramp_corridor_support_points.csv", ramp_points, ramp_fields)
    write_csv(output / "ramp_corridor_lidar_path.csv", ramp_path, [
        "time_sec", "path_arc_length_s_m", "independent_x_m", "independent_y_m",
        "independent_z_m", "matched_glim_index", "glim_xy_distance_m", "glim_z_m",
    ])
    write_json(output / "clearance_group_metrics.json", clearance_metrics)
    write_json(output / "map_landing_separation_blind.json", landing)

    plot_map_extraction(
        output / "ply_local_support_extraction_map.png", points, records, candidates,
        primary_results, primary_radius, int(config["plot"]["map_max_points"]),
    )
    plot_candidate_montage(
        output / "candidate_local_support_montage.png", points, candidates,
        primary_results, primary_radius, int(config["plot"]["montage_columns"]),
    )
    plot_three_way(output / "three_way_z_comparison.png", crosscheck)
    plot_clearance(output / "clearance_comparison.png", crosscheck)
    plot_independent_minus_mapping(output / "independent_minus_mapping_pose_z.png", crosscheck)
    plot_ramp_corridor(output / "ramp_corridor_s_z.png", ramp_points, ramp_path)

    blind_summary = build_blind_summary(
        config, primary_rows, crosscheck, associations, sensitivity, ramp_metrics,
        landing, before_blind,
    )
    blind_path = output / "blind_ply_surface_geometry_summary.json"
    write_json(blind_path, blind_summary)
    blind_digest = sha256_file(blind_path)
    (output / "blind_ply_surface_geometry_summary.sha256").write_text(
        f"{blind_digest}  blind_ply_surface_geometry_summary.json\n", encoding="utf-8"
    )

    # Phase B begins only after the blind JSON exists and its SHA256 is sealed above.
    physical_before = physical_gate.fingerprint(inputs["physical_gate_config"], include_hash=True)
    posthoc = posthoc_height(landing, inputs["physical_gate_config"])
    posthoc["blind_sha256"] = blind_digest
    plot_posthoc_height(output / "posthoc_measured_height.png", posthoc)

    after_blind = blind_input_fingerprints(inputs)
    physical_after = physical_gate.fingerprint(inputs["physical_gate_config"], include_hash=True)
    protected_unchanged = before_blind == after_blind and physical_before == physical_after
    if not protected_unchanged:
        raise RuntimeError("A protected input changed during the PLY support diagnostic")

    interpretation = determine_interpretation(blind_summary)
    final_summary = {
        "diagnostic_only": True,
        "diagnostic_name": "PLY local support-surface geometry cross-check",
        "measured_height_used_for_surface_selection": False,
        "measured_height_used_for_glim_association": False,
        "measured_height_used_for_blind_map_height": False,
        "fixed_labels": blind_summary["fixed_labels"],
        "primary_candidate_radius": primary_radius,
        "surface_success_count": blind_summary["surface_success_count"],
        "surface_ambiguous_count": blind_summary["surface_ambiguous_count"],
        "surface_failure_count": blind_summary["surface_failure_count"],
        "surface_failure_reasons": blind_summary["surface_failure_reasons"],
        "ground_surface_z_metrics": blind_summary["ground_surface_z_metrics"],
        "stage_surface_z_metrics": blind_summary["stage_surface_z_metrics"],
        "ground_independent_clearance_metrics": blind_summary[
            "ground_independent_clearance_metrics"
        ],
        "stage_independent_clearance_metrics": blind_summary[
            "stage_independent_clearance_metrics"
        ],
        "ground_mapping_clearance_metrics": blind_summary["ground_mapping_clearance_metrics"],
        "stage_mapping_clearance_metrics": blind_summary["stage_mapping_clearance_metrics"],
        "independent_minus_mapping_metrics": blind_summary["independent_minus_mapping_metrics"],
        "ramp_corridor_metrics": ramp_metrics,
        "blind_map_landing_separation": landing,
        "blind_sha256": blind_digest,
        "measured_stage_height_m": posthoc["measured_stage_height_m"],
        "posthoc_map_height_error": posthoc,
        "blind_analysis": blind_summary,
        "interpretation": interpretation,
        "historical_height_status_preserved": "HEIGHT_FAIL",
        "production_localization_failure": False,
        "protected_inputs_unchanged": protected_unchanged,
        "input_integrity": {
            "unchanged": protected_unchanged,
            "blind_inputs_before": before_blind,
            "blind_inputs_after": after_blind,
            "physical_config_phase_b_before": physical_before,
            "physical_config_phase_b_after": physical_after,
        },
        "input_counts": {
            "ply_vertices": int(points.shape[0]),
            "glim_mapping_trajectory_poses": int(trajectory.shape[0]),
            "independent_localization_rows": len(records),
            "independent_accepted_rows": sum(row["accepted"] for row in records),
            "diagnostic_candidates": len(candidates),
        },
        "output_directory": str(output),
    }
    write_json(output / "ply_surface_geometry_diagnostic_summary.json", final_summary)
    (output / "ply_surface_geometry_diagnostic_report.md").write_text(
        report_text(final_summary, crosscheck), encoding="utf-8"
    )
    return final_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    summary = run(args.config, args.output_directory)
    print(json.dumps({
        "surface_success_count": summary["surface_success_count"],
        "surface_ambiguous_count": summary["surface_ambiguous_count"],
        "surface_failure_count": summary["surface_failure_count"],
        "blind_map_landing_separation_m": summary[
            "blind_map_landing_separation"
        ]["map_landing_separation_m"],
        "posthoc_status": summary["posthoc_map_height_error"]["status"],
        "protected_inputs_unchanged": summary["protected_inputs_unchanged"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
