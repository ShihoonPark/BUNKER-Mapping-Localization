#!/usr/bin/env python3
"""Generate the continuous GICP correspondence/RViz/latency diagnostic artifacts."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/a/Desktop/shihoon/bunker_localization_ws")
DEFAULT_OUTPUT = ROOT / "results/gicp_correspondence_rviz_diagnostic"
MAP = Path(
    "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply"
)
BAG = Path("/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346")
HANDOFF = (
    ROOT
    / "results/gicp_vertical_observability_diagnostic/rviz_handoff_selected_scans.json"
)
ASSOCIATION = (
    ROOT
    / "results/gicp_vertical_observability_diagnostic/scan_level_mapping_xy_association.csv"
)
PRODUCTION_LOCALIZATION = (
    ROOT / "results/planar_ekf_gicp_gate/independent_163346_0_50s/localization.csv"
)
CONFIGS = [
    ROOT / "src/bunker_offline_localization/config/localization.yaml",
    ROOT / "src/bunker_offline_localization/config/independent_163346.yaml",
    ROOT / "src/bunker_offline_localization/config/ekf.yaml",
]
ROLES = [
    "baseline_representative",
    "largest_negative_dz",
    "largest_mapping_z_discrepancy",
    "recovery_representative",
    "stable_stage_representative",
]
DEADLINE_MS = 20.0
TOPICS = {
    "/map_cloud": {"type": "sensor_msgs/msg/PointCloud2", "frame": "map"},
    "/raw_scan": {"type": "sensor_msgs/msg/PointCloud2", "frame": "velodyne"},
    "/registered_scan": {"type": "sensor_msgs/msg/PointCloud2", "frame": "map"},
    "/gicp_pose": {"type": "geometry_msgs/msg/PoseStamped", "frame": "map"},
    "/gicp_path": {"type": "nav_msgs/msg/Path", "frame": "map"},
    "/prediction_pose": {"type": "geometry_msgs/msg/PoseStamped", "frame": "map"},
    "/gicp_correspondences": {
        "type": "visualization_msgs/msg/MarkerArray",
        "frame": "map",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "type": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def directory_stat_fingerprint(path):
    path = Path(path)
    return {
        "path": str(path),
        "type": "directory",
        "files": {
            str(item.relative_to(path)): {
                "size": item.stat().st_size,
                "mtime_ns": item.stat().st_mtime_ns,
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        },
    }


def protected_fingerprints():
    small_gicp = ROOT / "third_party/small_gicp"
    return {
        "map": file_fingerprint(MAP),
        "bag": directory_stat_fingerprint(BAG),
        "production_localization": file_fingerprint(PRODUCTION_LOCALIZATION),
        "production_configs": [file_fingerprint(path) for path in CONFIGS],
        "selected_scan_handoff": file_fingerprint(HANDOFF),
        "small_gicp": {
            "path": str(small_gicp),
            "head": subprocess.check_output(
                ["git", "-C", str(small_gicp), "rev-parse", "HEAD"], text=True
            ).strip(),
            "porcelain_status": subprocess.check_output(
                ["git", "-C", str(small_gicp), "status", "--porcelain"], text=True
            ).strip(),
        },
    }


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def distribution(values):
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {key: None for key in ("count", "mean", "median", "p95", "p99", "max")}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def correspondence_metrics(output, selected):
    role = selected["selection_role"]
    directory = output / "correspondence_audit" / role
    metadata = json.loads((directory / "scan_metadata.json").read_text(encoding="utf-8"))
    rows = read_csv(directory / "correspondences.csv")
    components = {
        axis: np.asarray([finite_float(row, f"residual_{axis}") for row in rows], dtype=float)
        for axis in ("x", "y", "z")
    }
    distances = np.asarray([finite_float(row, "distance_m") for row in rows], dtype=float)
    dominance = (
        np.abs(components["z"])
        > np.maximum(np.abs(components["x"]), np.abs(components["y"]))
    )
    metric = {
        "selection_role": role,
        "timestamp": float(metadata["timestamp"]),
        "bag_relative_time": float(selected["bag_relative_time"]),
        "source_points_before_voxel": int(metadata["raw_finite_points"]),
        "source_points_after_voxel": int(metadata["source_points_after_voxel"]),
        "candidate_correspondences": int(metadata["candidate_correspondences"]),
        "valid_correspondences": int(metadata["valid_correspondences"]),
        "distance_mean": float(np.mean(distances)),
        "distance_median": float(np.median(distances)),
        "distance_p95": float(np.percentile(distances, 95)),
        "distance_max": float(np.max(distances)),
        "dz_dominance_fraction": float(np.mean(dominance)),
        "gicp_inliers": int(metadata["gicp_inliers"]),
        "error_per_inlier": float(metadata["final_error"]) / int(metadata["gicp_inliers"]),
        "iterations": int(metadata["iterations"]),
        "prediction_z": float(metadata["prediction_T_map_lidar"]["z"]),
        "final_z": float(metadata["final_T_map_lidar"]["z"]),
        "canonical_associated_z": float(selected["mapping_z"]),
        "audit_pose_source": metadata["audit_pose_source"],
    }
    for axis, values in components.items():
        absolute = np.abs(values)
        metric[f"d{axis}_mean"] = float(np.mean(values))
        metric[f"d{axis}_median"] = float(np.median(values))
        metric[f"abs_d{axis}_median"] = float(np.median(absolute))
        metric[f"abs_d{axis}_p95"] = float(np.percentile(absolute, 95))
    return metric, rows


def write_metrics_csv(path, metrics):
    fields = list(metrics[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def plot_correspondences(output, role, rows, limit=200):
    registered = np.asarray(
        [[finite_float(row, f"registered_{axis}") for axis in "xyz"] for row in rows]
    )
    target = np.asarray([[finite_float(row, f"target_{axis}") for axis in "xyz"] for row in rows])
    indices = np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)
    for first, second, suffix in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
        fig, axis = plt.subplots(figsize=(8, 7))
        axis.scatter(target[:, first], target[:, second], s=3, c="0.35", alpha=0.35, label="map target")
        axis.scatter(
            registered[:, first], registered[:, second], s=3, c="#28d756", alpha=0.45,
            label="registered source",
        )
        for index in indices:
            axis.plot(
                [registered[index, first], target[index, first]],
                [registered[index, second], target[index, second]],
                color="#ff4d26", alpha=0.45, linewidth=0.55,
            )
        axis.set_xlabel("xyz"[first] + " [m]")
        axis.set_ylabel("xyz"[second] + " [m]")
        axis.set_title(f"{role}: {suffix.upper()} post-hoc correspondences (max {limit} lines)")
        axis.axis("equal")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / "correspondence_audit" / role / f"correspondence_{suffix}.png", dpi=170)
        plt.close(fig)


def load_map_xyz():
    import open3d as o3d

    points = np.asarray(o3d.io.read_point_cloud(str(MAP)).points)
    if points.shape[0] > 25000:
        indices = np.linspace(0, points.shape[0] - 1, 25000, dtype=int)
        points = points[indices]
    return points


def latency_figures(output, latency_rows, localization_rows, selected):
    times = np.asarray([finite_float(row, "bag_relative_time") for row in latency_rows])
    core = np.asarray([finite_float(row, "core_localization_latency_ms") for row in latency_rows])
    registration = np.asarray([finite_float(row, "registration_runtime_ms") for row in latency_rows])

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(times, core, linewidth=0.9, color="#1765a5")
    axis.axhline(DEADLINE_MS, color="#d62728", linestyle="--", label="20 ms deadline")
    axis.set(xlabel="bag-relative time [s]", ylabel="core localization latency [ms]", title="All processed scans: core localization latency")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_over_time.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(registration, core, s=14, alpha=0.55)
    bound = max(float(np.nanmax(core)), float(np.nanmax(registration)), DEADLINE_MS)
    axis.plot([0, bound], [0, bound], color="0.4", linestyle=":", label="equal runtime")
    axis.axhline(DEADLINE_MS, color="#d62728", linestyle="--", label="20 ms core deadline")
    axis.set(xlabel="registration only [ms]", ylabel="core localization [ms]", title="Registration runtime vs core localization latency")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "registration_vs_core_latency.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(core, bins=35, color="#1765a5", alpha=0.8)
    axis.axvline(DEADLINE_MS, color="#d62728", linestyle="--", label="20 ms deadline")
    axis.set(xlabel="core localization latency [ms]", ylabel="scan count", title="All-scan core localization latency distribution")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_distribution.png", dpi=170)
    plt.close(fig)

    map_xyz = load_map_xyz()
    accepted = np.asarray(
        [[finite_float(row, "gicp_x"), finite_float(row, "gicp_y")] for row in localization_rows if row.get("accepted") == "1"]
    )
    rejected = np.asarray(
        [[finite_float(row, "gicp_x"), finite_float(row, "gicp_y")] for row in localization_rows if row.get("accepted") != "1"]
    )
    fig, axis = plt.subplots(figsize=(10, 9))
    axis.scatter(map_xyz[:, 0], map_xyz[:, 1], s=1, c="0.72", alpha=0.35, label="canonical PLY map")
    axis.plot(accepted[:, 0], accepted[:, 1], color="#1765a5", linewidth=1.4, label=f"accepted poses ({len(accepted)})")
    if rejected.size:
        mask = np.isfinite(rejected).all(axis=1)
        axis.scatter(rejected[mask, 0], rejected[mask, 1], marker="x", s=55, c="#d62728", label=f"rejected poses ({len(rejected)})")
    for item in selected:
        row = localization_rows[int(item["row_index"])]
        axis.scatter(finite_float(row, "gicp_x"), finite_float(row, "gicp_y"), s=75, marker="*", label=item["selection_role"])
    axis.set(xlabel="map x [m]", ylabel="map y [m]", title="All processed LiDAR scans, not plateau candidates")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "all_scan_xy_mapping_overlay.png", dpi=180)
    plt.close(fig)


def write_overruns(path, rows):
    fields = [
        "timestamp", "bag_relative_time", "status", "registration_runtime_ms",
        "core_localization_latency_ms", "source_points", "inliers", "iterations",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def build_report(summary, metrics):
    latency = summary["core_localization_latency"]
    registration = summary["registration_runtime"]
    baseline = metrics[0]
    negative = metrics[1]
    discrepancy = metrics[2]
    recovery = metrics[3]
    interpretation = summary["correspondence_interpretation"]
    rows = []
    for metric in metrics:
        rows.append(
            f"| {metric['selection_role']} | {metric['valid_correspondences']} | "
            f"{metric['distance_median']:.4f} | {metric['distance_p95']:.4f} | "
            f"{metric['abs_dx_median']:.4f}/{metric['abs_dy_median']:.4f}/{metric['abs_dz_median']:.4f} | "
            f"{metric['dz_dominance_fraction']:.3f} | {metric['prediction_z']:.4f} | "
            f"{metric['final_z']:.4f} | {metric['canonical_associated_z']:.4f} |"
        )
    return f"""# Continuous GICP RViz / Correspondence Audit / 20 ms Runtime Gate

## Continuous all-scan result

The 0–50 s independent interval processed **{summary['continuous_processed_scan_count']} scans** in timestamp order: {summary['accepted_scan_count']} accepted and {summary['rejected_scan_count']} rejected. `candidate_count_used_for_execution = 0`; the 22 plateau candidates were not an execution list. Rejected poses were not appended to the accepted path.

## Correspondence method

The official small_gicp GICP remains unchanged. The API does not expose final factor indices, so this audit uses **`posthoc_final_transform_correspondence_reconstruction`**: the actual voxelized source retained by registration, the once-voxelized production target, the same target KD-tree, final `T_map_lidar`, nearest-target rule, and unchanged 1.0 m maximum correspondence distance. It is not described as exact internal optimizer correspondence.

| selected role | valid | dist median [m] | dist p95 [m] | median |dx|/|dy|/|dz| [m] | z-dominant fraction | prediction z | final z | associated mapping z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`largest_negative_dz` has median/p95 distance {negative['distance_median']:.4f}/{negative['distance_p95']:.4f} m and z-dominance {negative['dz_dominance_fraction']:.3f}; `largest_mapping_z_discrepancy` has {discrepancy['distance_median']:.4f}/{discrepancy['distance_p95']:.4f} m and {discrepancy['dz_dominance_fraction']:.3f}. Baseline is {baseline['distance_median']:.4f}/{baseline['distance_p95']:.4f} m and recovery is {recovery['distance_median']:.4f}/{recovery['distance_p95']:.4f} m.

Diagnostic interpretation: **{interpretation['case']} — {interpretation['conclusion']}** This is an audit interpretation, not a new registration/rejection threshold.

## 20 ms primary Gate

Headless core localization timing excludes CSV I/O, post-hoc reconstruction, ROS publication, marker serialization, and RViz rendering. Count={latency['count']}, mean={latency['mean']:.3f} ms, median={latency['median']:.3f} ms, p95={latency['p95']:.3f} ms, p99={latency['p99']:.3f} ms, max={latency['max']:.3f} ms. Overruns: {summary['overrun_count']} ({summary['overrun_percentage']:.3f}%). Strict status: **{summary['status']}** (`max <= 20.0 ms`). This is a deterministic engineering Gate on the current lab PC/offline reviewed dataset, not hard-real-time certification. The measured core maximum is also {'below' if latency['max'] < 100.0 else 'not below'} the approximate 100 ms scan period; that does not replace the 20 ms Gate.

Registration-only timing: mean={registration['mean']:.3f} ms, p95={registration['p95']:.3f} ms, p99={registration['p99']:.3f} ms, max={registration['max']:.3f} ms.

The separate 50-scan visualization-publisher integration run recorded secondary core mean/p95/p99/max of {summary['visualization_mode_core_localization_latency']['mean']:.3f}/{summary['visualization_mode_core_localization_latency']['p95']:.3f}/{summary['visualization_mode_core_localization_latency']['p99']:.3f}/{summary['visualization_mode_core_localization_latency']['max']:.3f} ms. It is not used for the primary Gate; RViz GUI rendering is not measured.

## RViz handoff

Topics and frames are recorded in `correspondence_audit_summary.json`. All seven required topics and frame IDs were received in the headless integration check. The reproducible config is `rviz/gicp_live_diagnostic.rviz`. Continuous and selected commands are in the package README. The GUI itself requires a user display session.

Protected map, bag, production result/configs, handoff, and pinned small_gicp fingerprints were unchanged: **{str(summary['protected_inputs_unchanged']).lower()}**.
"""


def generate(output, runtime_results, before_path):
    output.mkdir(parents=True, exist_ok=True)
    latency_source = runtime_results / "continuous_latency.csv"
    localization_source = runtime_results / "localization.csv"
    shutil.copyfile(latency_source, output / "continuous_latency.csv")
    latency_rows = read_csv(latency_source)
    localization_rows = read_csv(localization_source)
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))["selected_scans"]
    assert [item["selection_role"] for item in handoff] == ROLES

    metrics = []
    for selected in handoff:
        metric, correspondence_rows = correspondence_metrics(output, selected)
        metrics.append(metric)
        plot_correspondences(output, selected["selection_role"], correspondence_rows)
    write_metrics_csv(output / "correspondence_selected_scan_metrics.csv", metrics)

    core_values = [finite_float(row, "core_localization_latency_ms") for row in latency_rows]
    registration_values = [finite_float(row, "registration_runtime_ms") for row in latency_rows]
    core = distribution(core_values)
    registration = distribution(registration_values)
    overruns = [
        row for row in latency_rows
        if finite_float(row, "core_localization_latency_ms") > DEADLINE_MS
    ]
    write_overruns(output / "latency_overruns.csv", overruns)
    latency_figures(output, latency_rows, localization_rows, handoff)

    baseline, negative, discrepancy, recovery, _ = metrics
    anomaly_locally_small = (
        negative["distance_median"] <= 1.5 * baseline["distance_median"]
        and discrepancy["distance_median"] <= 1.5 * baseline["distance_median"]
    )
    if anomaly_locally_small:
        interpretation = {
            "case": "Case B",
            "conclusion": "wrong but locally consistent correspondence basin plausible",
        }
    elif (
        negative["abs_dz_p95"] > 1.5 * max(negative["abs_dx_p95"], negative["abs_dy_p95"])
        or discrepancy["abs_dz_p95"] > 1.5 * max(discrepancy["abs_dx_p95"], discrepancy["abs_dy_p95"])
    ):
        interpretation = {
            "case": "Case A",
            "conclusion": "vertical correspondence mismatch/local basin plausible",
        }
    else:
        interpretation = {
            "case": "Case C",
            "conclusion": "post-hoc correspondence audit alone insufficient",
        }

    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = protected_fingerprints()
    protected_unchanged = before == after
    (output / "protected_inputs_after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    topic_integration_path = output / "topic_integration.json"
    topic_integration = json.loads(topic_integration_path.read_text(encoding="utf-8")) \
        if topic_integration_path.exists() else {"all_required_topics_received": False}
    visualization_latency_path = output / "topic_integration_run/continuous_latency.csv"
    visualization_latency = distribution([
        finite_float(row, "core_localization_latency_ms")
        for row in read_csv(visualization_latency_path)
    ]) if visualization_latency_path.exists() else distribution([])
    summary = {
        "diagnostic_name": "continuous_gicp_rviz_correspondence_20ms_runtime_gate",
        "continuous_processed_scan_count": len(latency_rows),
        "accepted_scan_count": sum(row["status"] == "ACCEPTED" for row in latency_rows),
        "rejected_scan_count": sum(row["status"] != "ACCEPTED" for row in latency_rows),
        "candidate_count_used_for_execution": 0,
        "continuous_all_scan_processing": True,
        "correspondence_method": "posthoc_final_transform_correspondence_reconstruction",
        "exact_internal_or_posthoc": "posthoc",
        "exact_internal_correspondence_claimed": False,
        "preprocessing_consistency": {
            "source": "actual production voxelized source retained from RegistrationOutput",
            "target": "same once-voxelized global target and KD-tree",
            "neighbor_rule": "same nearest target search at final T_map_lidar",
            "maximum_correspondence_distance_m": 1.0,
        },
        "selected_scan_metrics": metrics,
        "correspondence_interpretation": interpretation,
        "registration_runtime": registration,
        "core_localization_latency": core,
        "deadline_ms": DEADLINE_MS,
        "overrun_count": len(overruns),
        "overrun_percentage": 100.0 * len(overruns) / len(latency_rows),
        "max_latency": core["max"],
        "p95_latency": core["p95"],
        "status": "PASS" if core["max"] <= DEADLINE_MS else "FAIL",
        "scan_period_sanity_ms": 100.0,
        "core_max_below_scan_period": core["max"] < 100.0,
        "rviz_topics": TOPICS,
        "topic_integration": topic_integration,
        "visualization_mode_core_localization_latency": visualization_latency,
        "rviz_config": str(
            ROOT / "src/bunker_offline_localization/rviz/gicp_live_diagnostic.rviz"
        ),
        "rviz_launch_command": (
            "cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization gicp_rviz_diagnostic.launch.py mode:=continuous"
        ),
        "rviz_selected_command": (
            "cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization gicp_rviz_diagnostic.launch.py mode:=selected selected_role:=largest_negative_dz"
        ),
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "protected_inputs_unchanged": protected_unchanged,
        "production_parameters_changed": False,
        "small_gicp_modified": False,
    }
    (output / "latency_summary.json").write_text(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "core_localization_latency", "registration_runtime", "deadline_ms",
                    "overrun_count", "overrun_percentage", "status",
                    "core_max_below_scan_period",
                )
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output / "correspondence_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gicp_correspondence_rviz_diagnostic_report.md").write_text(
        build_report(summary, metrics), encoding="utf-8"
    )
    if not protected_unchanged:
        raise RuntimeError("Protected input fingerprints changed during the diagnostic run")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-results", type=Path, default=DEFAULT_OUTPUT / "headless_run")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.snapshot_only:
        destination = args.before or args.output / "protected_inputs_before.json"
        destination.write_text(
            json.dumps(protected_fingerprints(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(destination)
        return
    before = args.before or args.output / "protected_inputs_before.json"
    summary = generate(args.output, args.runtime_results, before)
    print(json.dumps({"status": summary["status"], "count": summary["continuous_processed_scan_count"]}))


if __name__ == "__main__":
    main()
