#!/usr/bin/env python3

"""Diagnose plateau-label plausibility without changing the preserved height Gate."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_localization_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_imu_gicp_physical_gate as physical_gate

np = physical_gate.np
plt = physical_gate.plt
yaml = physical_gate.yaml


DIAGNOSTIC_CONCLUSION = (
    "height validation currently inconclusive because physical plateau labels "
    "are not independently established"
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_map_xy(path, maximum_points):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        raise ValueError(f"PLY map has no finite vertices: {path}")
    if maximum_points > 0 and points.shape[0] > maximum_points:
        indices = np.linspace(0, points.shape[0] - 1, maximum_points, dtype=int)
        points = points[indices]
    return points[:, :2]


def detect_full_interval_candidates(records, config):
    diagnostic = config["plateau_label_diagnostic"]
    if diagnostic.get("conclusion") != DIAGNOSTIC_CONCLUSION:
        raise ValueError("The diagnostic conclusion must remain the checked-in inconclusive statement")
    if diagnostic.get("measured_height_used_for_candidate_detection", False):
        raise ValueError("Measured height must not be used for diagnostic candidate detection")
    if diagnostic.get("assign_new_physical_labels", False):
        raise ValueError("Diagnostic-only search must not assign new physical surface labels")
    detection = dict(config["plateau_detection"])
    detection["search_start_sec"] = float(diagnostic["valid_start_sec"])
    detection["search_end_sec"] = float(diagnostic["valid_end_sec"])
    candidates = physical_gate.detect_plateaus(
        records, detection, config["pairing"]["max_pair_dt_sec"]
    )
    for candidate in candidates:
        candidate["diagnostic_candidate_id"] = candidate.pop("candidate_id")
        candidate["automatic_physical_label"] = "UNASSIGNED"
        candidate["used_for_height_selection"] = False
    return candidates, detection


def candidate_pair_height_diagnostics(candidates, measured_height):
    """Return natural ID-order post-hoc comparisons; never rank or select by measurement error."""
    rows = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            signed_difference = second["median_z_m"] - first["median_z_m"]
            absolute_difference = abs(signed_difference)
            absolute_error = abs(absolute_difference - measured_height)
            rows.append({
                "candidate_a_id": first["diagnostic_candidate_id"],
                "candidate_b_id": second["diagnostic_candidate_id"],
                "candidate_a_window_sec": (
                    f"{first['start_offset_sec']:.9f}-{first['end_offset_sec']:.9f}"
                ),
                "candidate_b_window_sec": (
                    f"{second['start_offset_sec']:.9f}-{second['end_offset_sec']:.9f}"
                ),
                "candidate_a_median_z_m": first["median_z_m"],
                "candidate_b_median_z_m": second["median_z_m"],
                "signed_z_b_minus_a_m": signed_difference,
                "absolute_pair_height_difference_m": absolute_difference,
                "measured_stage_height_m": measured_height,
                "post_hoc_absolute_error_m": absolute_error,
                "post_hoc_relative_error_percent": 100.0 * absolute_error / measured_height,
                "used_for_candidate_selection": False,
                "table_order": "ascending_candidate_id_not_measurement_error",
            })
    return rows


def historical_temporal_labeling(candidate_rows, height_validation):
    by_id = {int(row["candidate_id"]): row for row in candidate_rows}
    groups = height_validation["plateau_selection"]["groups"]
    stage_ids = [int(value) for value in groups["stage_top"]["candidate_ids"]]
    after_ids = [int(value) for value in groups["ground_after"]["candidate_ids"]]
    reported_ids = stage_ids + after_ids
    raw_z = {str(candidate_id): float(by_id[candidate_id]["median_z_m"]) for candidate_id in reported_ids}
    ordered_ids = sorted(reported_ids, key=lambda candidate_id: raw_z[str(candidate_id)])
    return {
        "preserved_height_status": height_validation["status"],
        "historical_stage_top_candidate_ids": stage_ids,
        "historical_ground_after_candidate_ids": after_ids,
        "raw_median_z_m_by_historical_candidate_id": raw_z,
        "raw_median_z_ascending_candidate_ids": ordered_ids,
        "raw_median_z_ordering_text": " < ".join(
            f"candidate {candidate_id} ({raw_z[str(candidate_id)]:.6f} m)"
            for candidate_id in ordered_ids
        ),
        "fitted_ground_plane_slope_deg": height_validation["ground_plane"]["slope_angle_deg"],
        "fitted_ground_plane_slope_m_per_m": height_validation["ground_plane"]["slope_magnitude"],
        "surface_labeling_warning": (
            "The 8.206 deg fitted ground-plane slope makes the prior physical surface labeling "
            "suspect; possible map tilt and localization-z inconsistency remain unresolved."
        ),
        "new_physical_labels_assigned": False,
    }


def plot_map_overlay(path, map_xy, records, candidates):
    accepted = [record for record in records if record["accepted"]]
    trajectory = np.asarray([record["position"][:2] for record in accepted])
    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[4.4, 1.2])
    axis = figure.add_subplot(grid[0, 0])
    annotation_axis = figure.add_subplot(grid[0, 1])
    axis.scatter(map_xy[:, 0], map_xy[:, 1], s=1.0, c="0.72", alpha=0.35, label="classroom PLY XY")
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=1.2, label="accepted GICP trajectory")
    centers = np.asarray([[candidate["median_x_m"], candidate["median_y_m"]] for candidate in candidates])
    axis.scatter(centers[:, 0], centers[:, 1], s=55, color="tab:red", zorder=4, label="stable candidates")
    colocated_start_offsets = {
        1: (-44, 14), 2: (0, 30), 3: (40, 14),
        4: (-22, -30), 5: (0, 22), 6: (0, -30),
    }
    for candidate in candidates:
        x = candidate["median_x_m"]
        y = candidate["median_y_m"]
        candidate_id = candidate["diagnostic_candidate_id"]
        odd = candidate_id % 2 == 1
        offset = colocated_start_offsets.get(
            candidate_id, ((7 if odd else -7), (7 if odd else -17))
        )
        axis.annotate(
            str(candidate_id), (x, y), xytext=offset,
            textcoords="offset points", ha=("left" if offset[0] >= 0 else "right"),
            fontsize=13, fontweight="bold", color="darkred",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            arrowprops=(
                {"arrowstyle": "-", "color": "darkred", "linewidth": 0.7, "alpha": 0.7}
                if candidate_id <= 6 else None
            ),
        )
    axis.set(
        xlabel="map x [m]", ylabel="map y [m]",
        title="0-50 s stable plateau candidates on classroom map (diagnostic IDs only)",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    annotation_axis.axis("off")
    annotation_axis.set_title("ID: start-end [s]", fontsize=12, loc="left")
    for index, candidate in enumerate(candidates):
        annotation_axis.text(
            0.0, 0.97 - index * 0.043,
            f"{candidate['diagnostic_candidate_id']:>2}: "
            f"{candidate['start_offset_sec']:5.1f}-{candidate['end_offset_sec']:5.1f}",
            transform=annotation_axis.transAxes, ha="left", va="top", fontsize=9,
            family="monospace",
        )
    annotation_axis.text(
        0.0, 0.0, "Diagnostic IDs only\n(no physical labels)",
        transform=annotation_axis.transAxes, ha="left", va="bottom", fontsize=10,
        fontweight="bold", color="darkred",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_synchronized_timeline(path, records, sensor, origin, candidates, start, end):
    accepted = [record for record in records if record["accepted"]]
    pose_times = np.asarray([record["offset"] for record in accepted])
    z = np.asarray([record["position"][2] for record in accepted])
    pitch_deg = np.degrees([record["rpy"][1] for record in accepted])
    imu_times = sensor["imu_times"] - origin
    odom_times = sensor["odom_times"] - origin
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    axes[0].plot(pose_times, z, color="tab:blue", linewidth=1.1)
    axes[0].set_ylabel("GICP z [m]")
    axes[1].plot(pose_times, pitch_deg, color="tab:orange", linewidth=1.1)
    axes[1].set_ylabel("GICP pitch [deg]")
    axes[2].plot(imu_times, sensor["gyro"][:, 1], color="tab:green", linewidth=0.8)
    axes[2].set_ylabel("IMU gyro y [rad/s]")
    axes[3].plot(odom_times, sensor["odom_linear_speed"], color="tab:purple", linewidth=0.9)
    axes[3].set_ylabel("odom speed [m/s]")
    axes[3].set_xlabel("bag-relative time [s]")
    colors = plt.get_cmap("tab20")
    for candidate in candidates:
        candidate_id = candidate["diagnostic_candidate_id"]
        color = colors((candidate_id - 1) % 20)
        for axis in axes:
            axis.axvspan(
                candidate["start_offset_sec"], candidate["end_offset_sec"],
                color=color, alpha=0.12,
            )
        midpoint = 0.5 * (candidate["start_offset_sec"] + candidate["end_offset_sec"])
        axes[0].text(
            midpoint, 0.96, str(candidate_id), transform=axes[0].get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold", color="black",
        )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(start, end)
    axes[0].set_title("Synchronized 0-50 s plateau-label diagnostic (IDs are not physical labels)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def report_text(summary, candidates):
    historical = summary["historical_temporal_labeling"]
    candidate_lines = "\n".join(
        f"| {candidate['diagnostic_candidate_id']} | {candidate['start_offset_sec']:.3f}-"
        f"{candidate['end_offset_sec']:.3f} | {candidate['sample_count']} | "
        f"{candidate['median_x_m']:.3f} | {candidate['median_y_m']:.3f} | "
        f"{candidate['median_z_m']:.4f} | {candidate['median_pitch_deg']:.3f} | UNASSIGNED |"
        for candidate in candidates
    )
    return f"""# Plateau physical-label diagnostic

## Diagnostic conclusion

**{summary['diagnostic_conclusion']}**

The preserved height result remains **{historical['preserved_height_status']}**, but this diagnostic
does not interpret it as a production-localization failure. It makes no localization, EKF,
small_gicp, map, voxel, registration-gate, bag, or prior-result changes.

## Height-blind candidate search

- Search interval: {summary['candidate_search_window_sec']} s
- Stable candidates: {summary['candidate_count']}
- Measured 0.150 m used for detection/selection: {summary['measured_height_used_for_candidate_detection']}
- New ground/stage labels assigned: {summary['new_physical_labels_assigned']}
- Detection criteria: absolute smoothed z rate, absolute smoothed pitch rate, minimum duration,
  consecutive accepted GICP rows, and the configured maximum timestamp gap.

| diagnostic ID | time window [s] | samples | median x | median y | median z | median pitch [deg] | physical label |
|---:|---|---:|---:|---:|---:|---:|---|
{candidate_lines}

`plateau_candidates_map_overlay.png` shows these diagnostic IDs on the classroom PLY and accepted
GICP trajectory. `plateau_candidates_timeline.png` synchronizes GICP z/pitch, raw IMU gyro-y,
odom speed, and the same candidate intervals.

## Preserved temporal labeling facts

The previous `temporal_partition_around_ramp_window` result labeled historical candidate 4 as
stage-top and candidates 5-7 as ground-after. Their raw median-z ordering is:

`{historical['raw_median_z_ordering_text']}`

This ordering is reported as an observation, not used to assign any new label. The previous fitted
ground-plane slope was {historical['fitted_ground_plane_slope_deg']:.3f} deg
({historical['fitted_ground_plane_slope_m_per_m']:.6f} m/m). **Diagnostic warning:**
{historical['surface_labeling_warning']}

## Post-hoc measured-height comparisons

`candidate_pair_height_diagnostics.csv` contains every candidate pair in natural candidate-ID
order, its signed and absolute raw median-z difference, and raw error against the measured
0.150 m. It is deliberately not sorted by measurement error, no closest pair is highlighted, and
every row states `used_for_candidate_selection=false`. The table is diagnostic-only and cannot
change the preserved HEIGHT_FAIL or create new ground/stage labels.

## Read-only verification

- Protected inputs unchanged: {summary['input_integrity']['unchanged']}
- Previous height result directory: `{summary['source_height_gate_directory']}`
- Output directory: `{summary['output_directory']}`
"""


def run(config_path, output_directory=None):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    diagnostic = config["plateau_label_diagnostic"]
    output = Path(output_directory or diagnostic["output_directory"]).resolve()
    source_height = Path(diagnostic["source_height_gate_directory"]).resolve()
    inputs = config["inputs"]
    protected = [
        inputs["bag"], Path(inputs["localization_csv"]).parent,
        inputs["map_ply"], inputs["glim_dump"], source_height,
        *inputs.get("preserved_result_directories", []),
    ]
    physical_gate.validate_output_path(output, protected)
    before = {
        "bag": physical_gate.fingerprint(inputs["bag"]),
        "localization_csv": physical_gate.fingerprint(inputs["localization_csv"], include_hash=True),
        "map_ply": physical_gate.fingerprint(inputs["map_ply"]),
        "source_height_gate": physical_gate.fingerprint(source_height),
    }
    output.mkdir(parents=True, exist_ok=True)
    origin = float(config["time_window"]["origin_timestamp"])
    start = float(diagnostic["valid_start_sec"])
    end = float(diagnostic["valid_end_sec"])
    records = physical_gate.read_localization_csv(
        inputs["localization_csv"], origin, start, end
    )
    candidates, detection = detect_full_interval_candidates(records, config)
    if not candidates:
        raise ValueError("No stable plateau candidates were detected in the diagnostic interval")
    sensor = physical_gate.bag_sensor_data(inputs["bag"], origin, start, end)
    map_xy = load_map_xy(inputs["map_ply"], int(diagnostic["map_max_plot_points"]))
    with (source_height / "height_validation.json").open(encoding="utf-8") as stream:
        height_validation = json.load(stream)
    historical_rows = read_csv(source_height / "candidate_plateaus.csv")
    historical = historical_temporal_labeling(historical_rows, height_validation)
    measured_height = float(config["physical_stage_height"]["height_m"])
    pair_rows = candidate_pair_height_diagnostics(candidates, measured_height)

    candidate_fields = [
        "diagnostic_candidate_id", "start_offset_sec", "end_offset_sec", "duration_sec",
        "sample_count", "median_x_m", "median_y_m", "median_z_m", "median_pitch_rad",
        "median_pitch_deg", "z_mad_m", "automatic_physical_label", "used_for_height_selection",
    ]
    physical_gate.write_csv(output / "diagnostic_plateau_candidates.csv", candidates, candidate_fields)
    physical_gate.write_csv(output / "candidate_pair_height_diagnostics.csv", pair_rows)
    plot_map_overlay(output / "plateau_candidates_map_overlay.png", map_xy, records, candidates)
    plot_synchronized_timeline(
        output / "plateau_candidates_timeline.png", records, sensor, origin, candidates, start, end
    )

    after = {
        "bag": physical_gate.fingerprint(inputs["bag"]),
        "localization_csv": physical_gate.fingerprint(inputs["localization_csv"], include_hash=True),
        "map_ply": physical_gate.fingerprint(inputs["map_ply"]),
        "source_height_gate": physical_gate.fingerprint(source_height),
    }
    integrity = {"unchanged": before == after, "before": before, "after": after}
    if not integrity["unchanged"]:
        raise RuntimeError("A protected diagnostic input changed")
    summary = {
        "diagnostic_name": "physical plateau-label diagnostic",
        "diagnostic_only": True,
        "candidate_search_window_sec": [start, end],
        "candidate_detection_parameters": detection,
        "candidate_count": len(candidates),
        "candidate_pair_count": len(pair_rows),
        "measured_stage_height_m": measured_height,
        "measured_height_used_for_candidate_detection": False,
        "new_physical_labels_assigned": False,
        "historical_temporal_labeling": historical,
        "preserved_height_status": historical["preserved_height_status"],
        "diagnostic_conclusion": DIAGNOSTIC_CONCLUSION,
        "source_height_gate_directory": str(source_height),
        "output_directory": str(output),
        "input_counts": {
            "localization_rows": len(records),
            "accepted_localization_rows": sum(record["accepted"] for record in records),
            "imu": int(sensor["imu_times"].size),
            "odom": int(sensor["odom_times"].size),
            "map_vertices_plotted": int(map_xy.shape[0]),
        },
        "input_integrity": integrity,
    }
    physical_gate.write_json(output / "plateau_label_diagnostic_summary.json", summary)
    (output / "plateau_label_diagnostic_report.md").write_text(
        report_text(summary, candidates), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    summary = run(args.config, args.output_directory)
    print(json.dumps({
        "candidate_count": summary["candidate_count"],
        "preserved_height_status": summary["preserved_height_status"],
        "diagnostic_conclusion": summary["diagnostic_conclusion"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
