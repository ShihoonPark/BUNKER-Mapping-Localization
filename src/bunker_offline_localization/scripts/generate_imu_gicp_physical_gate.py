#!/usr/bin/env python3

"""Read-only IMU/GICP physical-consistency analysis for the independent 0-50 s Gate."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bunker_localization_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


AXES = ("x", "y", "z")
EXPECTED_MOTION_NAMES = {"x": "roll", "y": "pitch", "z": "yaw"}


def skew(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def so3_exp(rotation_vector):
    vector = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(vector))
    generator = skew(vector)
    if theta < 1.0e-8:
        return np.eye(3) + generator + 0.5 * generator @ generator
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * generator + b * generator @ generator


def so3_log(rotation):
    """Stable principal SO(3) logarithm, including neighborhoods of zero and pi."""
    rotation = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    antisymmetric = vee((rotation - rotation.T) * 0.5)
    if theta < 1.0e-7:
        scale = 1.0 + theta * theta / 6.0
        return scale * antisymmetric
    if math.pi - theta < 1.0e-5:
        symmetric = 0.5 * (rotation + np.eye(3))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis /= np.linalg.norm(axis)
        if np.linalg.norm(antisymmetric) > 1.0e-10:
            if np.dot(axis, antisymmetric) < 0.0:
                axis = -axis
        else:
            largest = int(np.argmax(np.abs(axis)))
            if axis[largest] < 0.0:
                axis = -axis
        return theta * axis
    return theta / math.sin(theta) * antisymmetric


def quaternion_to_rotation(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Quaternion is non-finite or has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rotation_to_rpy(rotation):
    rotation = np.asarray(rotation, dtype=float)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw])


def median_absolute_deviation(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def robust_gyro_bias(gyro, mask=None):
    gyro = np.asarray(gyro, dtype=float)
    if mask is not None:
        gyro = gyro[np.asarray(mask, dtype=bool)]
    if gyro.ndim != 2 or gyro.shape[1] != 3 or gyro.shape[0] == 0:
        raise ValueError("At least one 3-axis stationary gyro sample is required")
    median = np.median(gyro, axis=0)
    return {
        "sample_count": int(gyro.shape[0]),
        "median_radps": median,
        "mean_radps": np.mean(gyro, axis=0),
        "mad_radps": np.median(np.abs(gyro - median), axis=0),
    }


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        stop = index + 1
        while stop < values.size and values[order[stop]] == values[order[index]]:
            stop += 1
        ranks[order[index:stop]] = 0.5 * (index + stop - 1) + 1.0
        index = stop
    return ranks


def pearson(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size < 3 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def spearman(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 3:
        return math.nan
    return pearson(rankdata(first[finite]), rankdata(second[finite]))


def sign_agreement(first, second, deadband=0.0):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    usable = np.isfinite(first) & np.isfinite(second)
    usable &= (np.abs(first) >= deadband) | (np.abs(second) >= deadband)
    if not np.any(usable):
        return math.nan, 0
    return float(np.mean(np.sign(first[usable]) == np.sign(second[usable]))), int(np.sum(usable))


def interval_time_weighted_average(times, values, start, end, max_sample_gap, boundary_gap):
    """Voronoi-cell time average using only samples inside an interval, without extrapolation."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if end <= start:
        return None
    selected = (times >= start) & (times <= end)
    sample_times = times[selected]
    sample_values = values[selected]
    if sample_times.size < 2:
        return None
    if sample_times[0] - start > boundary_gap or end - sample_times[-1] > boundary_gap:
        return None
    if np.max(np.diff(sample_times)) > max_sample_gap:
        return None
    edges = np.empty(sample_times.size + 1, dtype=float)
    edges[0] = start
    edges[-1] = end
    edges[1:-1] = 0.5 * (sample_times[:-1] + sample_times[1:])
    weights = np.diff(edges)
    if np.any(weights < 0.0):
        return None
    return np.sum(sample_values * weights[:, None], axis=0) / (end - start)


def read_localization_csv(path, origin_timestamp, start_sec, end_sec):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No localization rows in {path}")
    required = ["timestamp", "accepted", "gicp_x", "gicp_y", "gicp_z"]
    required += [f"gicp_q{name}" for name in ("x", "y", "z", "w")]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"Localization CSV is missing columns: {missing}")
    records = []
    for index, row in enumerate(rows):
        timestamp = float(row["timestamp"])
        offset = timestamp - origin_timestamp
        if offset < start_sec - 1.0e-9 or offset > end_sec + 1.0e-9:
            raise ValueError(f"Localization row {index} is outside [{start_sec}, {end_sec}] s")
        quaternion = np.asarray([float(row[f"gicp_q{name}"]) for name in ("x", "y", "z", "w")])
        records.append({
            "row_index": index,
            "timestamp": timestamp,
            "offset": offset,
            "accepted": row["accepted"] == "1",
            "position": np.asarray([float(row[f"gicp_{axis}"]) for axis in AXES]),
            "quaternion": quaternion,
            "quaternion_norm": float(np.linalg.norm(quaternion)),
            "rotation": quaternion_to_rotation(quaternion),
            "rpy": np.asarray([float(row[f"gicp_{name}"]) for name in ("roll", "pitch", "yaw")]),
        })
    return records


def build_gicp_pairs(records, max_pair_dt):
    pairs = []
    excluded = {"rejected_endpoint": 0, "nonpositive_dt": 0, "large_gap": 0}
    segment = 0
    for first, second in zip(records[:-1], records[1:]):
        dt = second["timestamp"] - first["timestamp"]
        reason = None
        if not first["accepted"] or not second["accepted"]:
            reason = "rejected_endpoint"
        elif dt <= 0.0:
            reason = "nonpositive_dt"
        elif dt > max_pair_dt:
            reason = "large_gap"
        if reason is not None:
            excluded[reason] += 1
            segment += 1
            continue
        relative = first["rotation"].T @ second["rotation"]
        pairs.append({
            "first_index": first["row_index"],
            "second_index": second["row_index"],
            "start_timestamp": first["timestamp"],
            "end_timestamp": second["timestamp"],
            "start_offset": first["offset"],
            "end_offset": second["offset"],
            "mid_offset": 0.5 * (first["offset"] + second["offset"]),
            "dt": dt,
            "segment": segment,
            "omega_gicp": so3_log(relative) / dt,
        })
    return pairs, excluded


def bag_sensor_data(bag_path, origin_timestamp, start_sec, end_sec):
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from sensor_msgs.msg import Imu

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    reader.set_filter(StorageFilter(topics=["/imu/data", "/odom"]))
    imu_times = []
    gyro = []
    imu_orientation_identity = 0
    odom_times = []
    odom_linear_speed = []
    odom_angular_speed = []
    limit_timestamp_ns = int((origin_timestamp + end_sec + 1.0) * 1.0e9)
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if bag_timestamp_ns > limit_timestamp_ns:
            break
        if topic == "/imu/data":
            message = deserialize_message(serialized, Imu)
            timestamp = message.header.stamp.sec + message.header.stamp.nanosec * 1.0e-9
            offset = timestamp - origin_timestamp
            if start_sec <= offset <= end_sec:
                imu_times.append(timestamp)
                gyro.append([
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ])
                orientation = message.orientation
                if (
                    orientation.x == 0.0 and orientation.y == 0.0 and
                    orientation.z == 0.0 and orientation.w == 1.0
                ):
                    imu_orientation_identity += 1
        elif topic == "/odom":
            message = deserialize_message(serialized, Odometry)
            timestamp = message.header.stamp.sec + message.header.stamp.nanosec * 1.0e-9
            offset = timestamp - origin_timestamp
            if start_sec <= offset <= end_sec:
                linear = message.twist.twist.linear
                angular = message.twist.twist.angular
                odom_times.append(timestamp)
                odom_linear_speed.append(math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2))
                odom_angular_speed.append(math.sqrt(angular.x ** 2 + angular.y ** 2 + angular.z ** 2))
    return {
        "imu_times": np.asarray(imu_times),
        "gyro": np.asarray(gyro),
        "imu_orientation_identity_count": imu_orientation_identity,
        "odom_times": np.asarray(odom_times),
        "odom_linear_speed": np.asarray(odom_linear_speed),
        "odom_angular_speed": np.asarray(odom_angular_speed),
    }


def nearest_values(query_times, sample_times, sample_values):
    query_times = np.asarray(query_times)
    sample_times = np.asarray(sample_times)
    if sample_times.size == 0:
        raise ValueError("No source samples for nearest-neighbor association")
    right = np.searchsorted(sample_times, query_times)
    right = np.clip(right, 0, sample_times.size - 1)
    left = np.clip(right - 1, 0, sample_times.size - 1)
    choose_right = np.abs(sample_times[right] - query_times) < np.abs(sample_times[left] - query_times)
    indices = np.where(choose_right, right, left)
    return np.asarray(sample_values)[indices], np.abs(sample_times[indices] - query_times)


def select_stationary_bias(sensor, origin, config):
    times = sensor["imu_times"]
    offsets = times - origin
    linear, linear_dt = nearest_values(times, sensor["odom_times"], sensor["odom_linear_speed"])
    angular, angular_dt = nearest_values(times, sensor["odom_times"], sensor["odom_angular_speed"])
    association_dt = np.maximum(linear_dt, angular_dt)
    gyro_norm = np.linalg.norm(sensor["gyro"], axis=1)
    mask = (offsets >= config["candidate_start_sec"]) & (offsets <= config["candidate_end_sec"])
    candidate_count = int(np.count_nonzero(mask))
    mask &= association_dt <= config["max_odom_association_dt_sec"]
    mask &= linear <= config["max_odom_linear_speed_mps"]
    mask &= angular <= config["max_odom_angular_speed_radps"]
    mask &= gyro_norm <= config["max_gyro_norm_radps"]
    result = robust_gyro_bias(sensor["gyro"], mask)
    result.update({
        "candidate_window_sec": [config["candidate_start_sec"], config["candidate_end_sec"]],
        "candidate_sample_count": candidate_count,
        "selection_thresholds": dict(config),
        "selected_first_offset_sec": float(np.min(offsets[mask])),
        "selected_last_offset_sec": float(np.max(offsets[mask])),
        "selected_odom_linear_speed_max_mps": float(np.max(linear[mask])),
        "selected_odom_angular_speed_max_radps": float(np.max(angular[mask])),
        "selected_gyro_norm_max_radps": float(np.max(gyro_norm[mask])),
    })
    return result, mask


def attach_imu_averages(pairs, sensor, bias, interval_config, lidar_R_imu):
    usable = []
    corrected = sensor["gyro"] - np.asarray(bias)[None, :]
    analysis_segment = 0
    previous_pair = None
    for pair in pairs:
        average_imu = interval_time_weighted_average(
            sensor["imu_times"], corrected, pair["start_timestamp"], pair["end_timestamp"],
            interval_config["max_sample_gap_sec"], interval_config["max_boundary_gap_sec"],
        )
        if average_imu is None:
            analysis_segment += 1
            previous_pair = None
            continue
        if previous_pair is not None and (
            pair["first_index"] != previous_pair["second_index"] or
            pair["segment"] != previous_pair["segment"]
        ):
            analysis_segment += 1
        enriched = dict(pair)
        enriched["segment"] = analysis_segment
        enriched["omega_imu"] = lidar_R_imu @ average_imu
        usable.append(enriched)
        previous_pair = pair
    return usable, corrected


def metric_row(window, axis, times, gicp, imu, deadband):
    delta = imu - gicp
    agreement, agreement_count = sign_agreement(gicp, imu, deadband)
    gicp_peak_index = int(np.argmax(np.abs(gicp)))
    imu_peak_index = int(np.argmax(np.abs(imu)))
    return {
        "window": window,
        "axis": axis,
        "motion_name": EXPECTED_MOTION_NAMES[axis],
        "usable_samples": int(gicp.size),
        "pearson": pearson(gicp, imu),
        "spearman": spearman(gicp, imu),
        "rmse_radps": float(np.sqrt(np.mean(delta * delta))),
        "mae_radps": float(np.mean(np.abs(delta))),
        "sign_agreement": agreement,
        "sign_agreement_samples": agreement_count,
        "gicp_std_radps": float(np.std(gicp)),
        "imu_std_radps": float(np.std(imu)),
        "gicp_peak_abs_radps": float(abs(gicp[gicp_peak_index])),
        "gicp_peak_offset_sec": float(times[gicp_peak_index]),
        "imu_peak_abs_radps": float(abs(imu[imu_peak_index])),
        "imu_peak_offset_sec": float(times[imu_peak_index]),
    }


def compute_axis_metrics(pairs, ramp_start, ramp_end, deadband):
    rows = []
    arrays = {}
    for window, predicate in (
        ("full_0_50s", lambda pair: True),
        ("ramp_25_35s", lambda pair: pair["start_offset"] >= ramp_start and pair["end_offset"] <= ramp_end),
    ):
        selected = [pair for pair in pairs if predicate(pair)]
        times = np.asarray([pair["mid_offset"] for pair in selected])
        gicp = np.asarray([pair["omega_gicp"] for pair in selected])
        imu = np.asarray([pair["omega_imu"] for pair in selected])
        segments = np.asarray([pair["segment"] for pair in selected], dtype=int)
        arrays[window] = {"times": times, "gicp": gicp, "imu": imu, "segments": segments}
        for index, axis in enumerate(AXES):
            rows.append(metric_row(window, axis, times, gicp[:, index], imu[:, index], deadband))
    return rows, arrays


def compute_axis_correlation_matrices(arrays):
    rows = []
    for window, values in arrays.items():
        correlations = np.empty((3, 3))
        for gicp_index, gicp_axis in enumerate(AXES):
            for imu_index, imu_axis in enumerate(AXES):
                correlation = pearson(values["gicp"][:, gicp_index], values["imu"][:, imu_index])
                correlations[gicp_index, imu_index] = correlation
                rows.append({
                    "window": window,
                    "gicp_axis": gicp_axis,
                    "imu_axis": imu_axis,
                    "pearson": correlation,
                    "spearman": spearman(
                        values["gicp"][:, gicp_index], values["imu"][:, imu_index]
                    ),
                    "expected_mapping": gicp_axis == imu_axis,
                    "best_absolute_mapping": False,
                })
        for gicp_index in range(3):
            best = int(np.nanargmax(np.abs(correlations[gicp_index])))
            for row in rows:
                if row["window"] == window and row["gicp_axis"] == AXES[gicp_index] and row["imu_axis"] == AXES[best]:
                    row["best_absolute_mapping"] = True
    return rows


def lag_correlation(times, reference, signal, segments, lag):
    compared_reference = []
    compared_signal = []
    for segment in np.unique(segments):
        mask = segments == segment
        segment_times = times[mask]
        if segment_times.size < 3:
            continue
        query = segment_times + lag
        inside = (query >= segment_times[0]) & (query <= segment_times[-1])
        if not np.any(inside):
            continue
        compared_reference.extend(reference[mask][inside])
        compared_signal.extend(np.interp(query[inside], segment_times, signal[mask]))
    return pearson(compared_reference, compared_signal), len(compared_reference)


def search_lags(times, reference, signal, segments, minimum, maximum, step):
    count = int(round((maximum - minimum) / step))
    lags = minimum + np.arange(count + 1) * step
    results = []
    for lag in lags:
        correlation, samples = lag_correlation(times, reference, signal, segments, float(lag))
        results.append({"lag_sec": float(lag), "pearson": correlation, "usable_samples": samples})
    finite = [index for index, row in enumerate(results) if np.isfinite(row["pearson"])]
    if not finite:
        return results, None, None
    best = max(finite, key=lambda index: abs(results[index]["pearson"]))
    zero = min(finite, key=lambda index: abs(results[index]["lag_sec"]))
    return results, best, zero


def compute_lag_rows(arrays, config):
    rows = []
    summary = {}
    for window, values in arrays.items():
        summary[window] = {}
        for index, axis in enumerate(AXES):
            lag_rows, best, zero = search_lags(
                values["times"], values["gicp"][:, index], values["imu"][:, index],
                values["segments"], config["minimum_sec"], config["maximum_sec"], config["step_sec"],
            )
            for row_index, row in enumerate(lag_rows):
                row.update({
                    "window": window,
                    "axis": axis,
                    "is_zero_lag": row_index == zero,
                    "is_best_absolute_correlation": row_index == best,
                })
                rows.append(row)
            summary[window][axis] = {
                "zero_lag_sec": None if zero is None else lag_rows[zero]["lag_sec"],
                "zero_lag_pearson": None if zero is None else lag_rows[zero]["pearson"],
                "best_lag_sec": None if best is None else lag_rows[best]["lag_sec"],
                "best_lag_pearson": None if best is None else lag_rows[best]["pearson"],
                "best_lag_usable_samples": None if best is None else lag_rows[best]["usable_samples"],
            }
    return rows, summary


def interpolate_vector(times, values, timestamp, max_gap):
    right = int(np.searchsorted(times, timestamp))
    if right == 0:
        if abs(times[0] - timestamp) <= max_gap:
            return values[0]
        raise ValueError("No IMU sample before integration boundary")
    if right == times.size:
        if abs(timestamp - times[-1]) <= max_gap:
            return values[-1]
        raise ValueError("No IMU sample after integration boundary")
    left = right - 1
    dt = times[right] - times[left]
    if dt <= 0.0 or dt > max_gap:
        raise ValueError("IMU gap exceeds integration limit")
    ratio = (timestamp - times[left]) / dt
    return (1.0 - ratio) * values[left] + ratio * values[right]


def integrate_gyro_interval(times, gyro, start, end, max_gap):
    if end <= start:
        raise ValueError("Gyro integration interval must have positive duration")
    inside = times[(times > start) & (times < end)]
    knots = np.concatenate(([start], inside, [end]))
    rotation = np.eye(3)
    for first, second in zip(knots[:-1], knots[1:]):
        if second - first > max_gap:
            raise ValueError("IMU gap exceeds integration limit")
        first_gyro = interpolate_vector(times, gyro, first, max_gap)
        second_gyro = interpolate_vector(times, gyro, second, max_gap)
        rotation = rotation @ so3_exp(0.5 * (first_gyro + second_gyro) * (second - first))
    return rotation


def integrate_constant_rate(rate, duration, step=0.005):
    times = np.arange(0.0, duration + 0.5 * step, step)
    if times[-1] < duration:
        times = np.append(times, duration)
    values = np.repeat(np.asarray(rate, dtype=float)[None, :], times.size, axis=0)
    return integrate_gyro_interval(times, values, 0.0, duration, 2.0 * step)


def orientation_comparison(records, sensor_times, corrected_gyro, lidar_R_imu, ramp_start, ramp_end, max_gap):
    accepted = [record for record in records if record["accepted"] and ramp_start <= record["offset"] <= ramp_end]
    if len(accepted) < 2:
        raise ValueError("Ramp window needs at least two accepted GICP orientations")
    anchor = accepted[0]
    imu_rotation = anchor["rotation"].copy()
    previous_time = anchor["timestamp"]
    rows = []
    for record in accepted:
        usable = True
        if record is not anchor:
            try:
                body_delta = integrate_gyro_interval(
                    sensor_times, corrected_gyro, previous_time, record["timestamp"], max_gap
                )
                imu_rotation = imu_rotation @ lidar_R_imu @ body_delta @ lidar_R_imu.T
            except ValueError:
                usable = False
        if usable:
            gicp_relative = anchor["rotation"].T @ record["rotation"]
            imu_relative = anchor["rotation"].T @ imu_rotation
            error = gicp_relative.T @ imu_relative
            gicp_rpy = rotation_to_rpy(gicp_relative)
            imu_rpy = rotation_to_rpy(imu_relative)
            rows.append({
                "timestamp": record["timestamp"],
                "offset_sec": record["offset"],
                "gicp_relative_roll_rad": gicp_rpy[0],
                "gicp_relative_pitch_rad": gicp_rpy[1],
                "gicp_relative_yaw_rad": gicp_rpy[2],
                "imu_relative_roll_rad": imu_rpy[0],
                "imu_relative_pitch_rad": imu_rpy[1],
                "imu_relative_yaw_rad": imu_rpy[2],
                "rotation_angle_error_rad": np.linalg.norm(so3_log(error)),
                "usable": True,
            })
        previous_time = record["timestamp"]
    if len(rows) < 2:
        raise ValueError("No usable ramp orientation integration")
    final_gicp = np.asarray([
        rows[-1][f"gicp_relative_{axis}_rad"] for axis in ("roll", "pitch", "yaw")
    ])
    final_imu = np.asarray([
        rows[-1][f"imu_relative_{axis}_rad"] for axis in ("roll", "pitch", "yaw")
    ])
    errors_deg = np.degrees([row["rotation_angle_error_rad"] for row in rows])
    summary = {
        "anchor_offset_sec": rows[0]["offset_sec"],
        "last_offset_sec": rows[-1]["offset_sec"],
        "usable_orientations": len(rows),
        "rotation_error_mean_deg": float(np.mean(errors_deg)),
        "rotation_error_p95_deg": float(np.percentile(errors_deg, 95)),
        "rotation_error_max_deg": float(np.max(errors_deg)),
        "final_rotation_error_deg": float(errors_deg[-1]),
        "gicp_net_relative_rpy_deg": np.degrees(final_gicp).tolist(),
        "imu_net_relative_rpy_deg": np.degrees(final_imu).tolist(),
        "gicp_net_relative_rotation_vector_deg": np.degrees(so3_log(gicp_relative)).tolist(),
        "imu_net_relative_rotation_vector_deg": np.degrees(so3_log(imu_relative)).tolist(),
        "gicp_net_relative_rotation_angle_deg": float(np.degrees(np.linalg.norm(so3_log(gicp_relative)))),
        "imu_net_relative_rotation_angle_deg": float(np.degrees(np.linalg.norm(so3_log(imu_relative)))),
    }
    return rows, summary


def robust_plane_fit(points, maximum_iterations=30):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("At least three xyz points are required for plane fitting")
    design = np.column_stack((points[:, 0], points[:, 1], np.ones(points.shape[0])))
    weights = np.ones(points.shape[0])
    coefficients = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
    for iteration in range(maximum_iterations):
        residuals = points[:, 2] - design @ coefficients
        scale = 1.4826 * median_absolute_deviation(residuals)
        if not np.isfinite(scale) or scale < 1.0e-9:
            break
        normalized = np.abs(residuals) / (1.345 * scale)
        weights = np.ones_like(normalized)
        outliers = normalized > 1.0
        weights[outliers] = 1.0 / normalized[outliers]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_z = points[:, 2] * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)[0]
        if np.linalg.norm(updated - coefficients) < 1.0e-10:
            coefficients = updated
            break
        coefficients = updated
    residuals = points[:, 2] - design @ coefficients
    return {
        "coefficients": coefficients,
        "residuals": residuals,
        "robust_spread_m": 1.4826 * median_absolute_deviation(residuals),
        "iterations": iteration + 1,
    }


def estimate_height_from_points(ground_before, stage_top, ground_after, physical, bootstrap_samples=0, seed=0):
    before = np.asarray(ground_before, dtype=float)
    stage = np.asarray(stage_top, dtype=float)
    after = np.asarray(ground_after, dtype=float)
    ground = np.vstack((before, after))
    plane = robust_plane_fit(ground)
    a, b, c = plane["coefficients"]

    def residual(points):
        return points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)

    before_residual = residual(before)
    stage_residual = residual(stage)
    after_residual = residual(after)
    estimated = float(np.median(stage_residual))
    robust_spread = float(1.4826 * median_absolute_deviation(stage_residual))
    before_height = float(np.median(stage_residual) - np.median(before_residual))
    after_height = float(np.median(stage_residual) - np.median(after_residual))
    bootstrap = []
    generator = np.random.default_rng(seed)
    for _ in range(int(bootstrap_samples)):
        sampled_before = before[generator.integers(0, before.shape[0], before.shape[0])]
        sampled_after = after[generator.integers(0, after.shape[0], after.shape[0])]
        sampled_stage = stage[generator.integers(0, stage.shape[0], stage.shape[0])]
        try:
            sampled_plane = robust_plane_fit(np.vstack((sampled_before, sampled_after)))
        except (ValueError, np.linalg.LinAlgError):
            continue
        sa, sb, sc = sampled_plane["coefficients"]
        sampled_residual = sampled_stage[:, 2] - (
            sa * sampled_stage[:, 0] + sb * sampled_stage[:, 1] + sc
        )
        bootstrap.append(float(np.median(sampled_residual)))
    confidence = [None, None]
    if bootstrap:
        confidence = [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]
    tolerance = float(physical["tolerance_m"])
    consistency_tolerance = float(physical.get("before_after_consistency_tolerance_m", tolerance))
    slope_warning_threshold = float(physical.get("ground_plane_slope_warning_threshold", 0.10))
    before_after_difference = abs(before_height - after_height)
    slope_magnitude = float(math.hypot(a, b))
    result = {
        "estimated_stage_height_m": estimated,
        "robust_spread_m": robust_spread,
        "bootstrap_95pct_ci_m": confidence,
        "ground_plane": {
            "a_dz_dx": float(a),
            "b_dz_dy": float(b),
            "c_m": float(c),
            "slope_magnitude": slope_magnitude,
            "slope_angle_deg": float(math.degrees(math.atan(slope_magnitude))),
            "robust_spread_m": float(plane["robust_spread_m"]),
            "slope_warning_threshold": slope_warning_threshold,
            "slope_warning": slope_magnitude > slope_warning_threshold,
        },
        "sample_count": {
            "ground_before": int(before.shape[0]),
            "stage_top": int(stage.shape[0]),
            "ground_after": int(after.shape[0]),
        },
        "ground_before_estimate_m": before_height,
        "ground_after_estimate_m": after_height,
        "before_reference_height_m": before_height,
        "after_reference_height_m": after_height,
        "before_after_difference_m": before_after_difference,
        "before_after_consistency": {
            "difference_m": before_after_difference,
            "tolerance_m": consistency_tolerance,
            "consistent": before_after_difference <= consistency_tolerance,
            "status": "CONSISTENT" if before_after_difference <= consistency_tolerance else "INCONSISTENT",
        },
        "physical_stage_height_available": bool(physical["available"]),
        "physical_stage_height_m": float(physical["height_m"]) if physical["available"] else None,
        "measured_stage_height_m": float(physical["height_m"]) if physical["available"] else None,
        "measurement_method": physical.get("measurement_method"),
        "measurement_reference": physical.get("measurement_reference"),
        "tolerance_m": tolerance,
        "tolerance_interpretation": physical.get(
            "tolerance_interpretation", "initial_screening_threshold_not_calibrated_accuracy"
        ),
    }
    if not physical["available"]:
        result.update({
            "status": "HEIGHT_PENDING", "absolute_error_m": None,
            "relative_error": None, "relative_error_percent": None,
        })
    else:
        absolute_error = abs(estimated - float(physical["height_m"]))
        relative_error = absolute_error / abs(float(physical["height_m"])) if physical["height_m"] else None
        if absolute_error > tolerance:
            status = "HEIGHT_FAIL"
        elif before_after_difference > consistency_tolerance or slope_magnitude > slope_warning_threshold:
            status = "HEIGHT_WARN"
        else:
            status = "HEIGHT_PASS"
        result.update({
            "status": status,
            "absolute_error_m": absolute_error,
            "relative_error": relative_error,
            "relative_error_percent": 100.0 * relative_error if relative_error is not None else None,
        })
    return result


def detect_plateaus(records, config, max_pair_dt):
    accepted = [
        record for record in records if record["accepted"] and
        config["search_start_sec"] <= record["offset"] <= config["search_end_sec"]
    ]
    contiguous = []
    for record in accepted:
        if (
            not contiguous or record["row_index"] != contiguous[-1][-1]["row_index"] + 1 or
            record["timestamp"] - contiguous[-1][-1]["timestamp"] > max_pair_dt
        ):
            contiguous.append([record])
        else:
            contiguous[-1].append(record)
    stable_intervals = []
    width = max(1, int(config.get("smoothing_window_samples", 1)))
    half = width // 2
    for segment in contiguous:
        if len(segment) < 2:
            continue
        z = np.asarray([record["position"][2] for record in segment])
        pitch = np.unwrap([record["rpy"][1] for record in segment])
        smooth_z = np.asarray([
            np.median(z[max(0, index - half):min(len(z), index + half + 1)])
            for index in range(len(z))
        ])
        smooth_pitch = np.asarray([
            np.median(pitch[max(0, index - half):min(len(pitch), index + half + 1)])
            for index in range(len(pitch))
        ])
        for index, (first, second) in enumerate(zip(segment[:-1], segment[1:])):
            dt = second["timestamp"] - first["timestamp"]
            z_rate = (smooth_z[index + 1] - smooth_z[index]) / dt
            pitch_rate = (smooth_pitch[index + 1] - smooth_pitch[index]) / dt
            if abs(z_rate) <= config["max_abs_z_rate_mps"] and abs(pitch_rate) <= config["max_abs_pitch_rate_radps"]:
                stable_intervals.append((first, second))
    groups = []
    for interval in stable_intervals:
        if not groups or interval[0]["row_index"] != groups[-1][-1][1]["row_index"]:
            groups.append([interval])
        else:
            groups[-1].append(interval)
    candidates = []
    for group in groups:
        pose_records = [group[0][0]] + [interval[1] for interval in group]
        duration = pose_records[-1]["offset"] - pose_records[0]["offset"]
        if duration < config["minimum_duration_sec"] or len(pose_records) < config["minimum_samples"]:
            continue
        positions = np.asarray([record["position"] for record in pose_records])
        pitches = np.asarray([record["rpy"][1] for record in pose_records])
        candidates.append({
            "candidate_id": len(candidates) + 1,
            "start_offset_sec": pose_records[0]["offset"],
            "end_offset_sec": pose_records[-1]["offset"],
            "duration_sec": duration,
            "sample_count": len(pose_records),
            "median_x_m": float(np.median(positions[:, 0])),
            "median_y_m": float(np.median(positions[:, 1])),
            "median_z_m": float(np.median(positions[:, 2])),
            "median_pitch_rad": float(np.median(pitches)),
            "median_pitch_deg": float(np.degrees(np.median(pitches))),
            "z_mad_m": median_absolute_deviation(positions[:, 2]),
        })
    return candidates


def selected_window_points(records, window):
    if window is None:
        return np.empty((0, 3))
    start, end = window
    return np.asarray([
        record["position"] for record in records
        if record["accepted"] and start <= record["offset"] <= end
    ])


def classify_plateau_candidates(candidates, ramp_start, ramp_end, selection_config):
    """Partition pre-detected candidates without receiving or inspecting the measured height."""
    if selection_config.get("measured_height_used_for_selection", False):
        raise ValueError("Measured stage height must never participate in plateau selection")
    groups = {"ground_before": [], "stage_top": [], "ground_after": []}
    boundary_candidates = []
    for candidate in candidates:
        start = candidate["start_offset_sec"]
        end = candidate["end_offset_sec"]
        if end <= ramp_start:
            group = "ground_before"
        elif start >= ramp_start and end <= ramp_end:
            group = "stage_top"
        elif start >= ramp_end:
            group = "ground_after"
        else:
            group = "ambiguous_boundary_overlap"
            boundary_candidates.append(candidate["candidate_id"])
        candidate["temporal_group"] = group
        candidate["selected_for_height"] = False
        if group in groups:
            groups[group].append(candidate)
    ambiguity_reasons = []
    if boundary_candidates:
        ambiguity_reasons.append(f"candidates overlap temporal boundaries: {boundary_candidates}")
    if not groups["ground_before"]:
        ambiguity_reasons.append("no stable candidate before the ramp window")
    if len(groups["stage_top"]) != 1:
        ambiguity_reasons.append(
            f"expected exactly one stable candidate inside the ramp window, found {len(groups['stage_top'])}"
        )
    if not groups["ground_after"]:
        ambiguity_reasons.append("no stable candidate after the ramp window")
    ambiguous = bool(ambiguity_reasons)
    if not ambiguous:
        for group_candidates in groups.values():
            for candidate in group_candidates:
                candidate["selected_for_height"] = True
    group_summary = {}
    for name, group_candidates in groups.items():
        group_summary[name] = {
            "candidate_ids": [candidate["candidate_id"] for candidate in group_candidates],
            "time_windows_sec": [
                [candidate["start_offset_sec"], candidate["end_offset_sec"]]
                for candidate in group_candidates
            ],
            "candidate_sample_counts": [candidate["sample_count"] for candidate in group_candidates],
            "total_samples": int(sum(candidate["sample_count"] for candidate in group_candidates)),
        }
    return {
        "method": selection_config["method"],
        "surface_identity_basis": "predeclared_ground_to_ramp_to_ground_temporal_sequence",
        "external_surface_label_confirmation": False,
        "ramp_window_sec": [ramp_start, ramp_end],
        "measured_height_used_for_selection": False,
        "selection_completed_before_physical_comparison": True,
        "ambiguous": ambiguous,
        "ambiguity_reasons": ambiguity_reasons,
        "groups": group_summary,
    }


def points_for_candidate_group(records, candidates, candidate_ids):
    selected = []
    candidate_ids = set(candidate_ids)
    for candidate in candidates:
        if candidate["candidate_id"] not in candidate_ids:
            continue
        points = selected_window_points(
            records, [candidate["start_offset_sec"], candidate["end_offset_sec"]]
        )
        if points.size:
            selected.append(points)
    return np.vstack(selected) if selected else np.empty((0, 3))


def incomplete_height_result(physical, status, reason, selection=None):
    measured = float(physical["height_m"]) if physical["available"] else None
    return {
        "status": status,
        "estimated_stage_height_m": None,
        "measured_stage_height_m": measured,
        "physical_stage_height_available": bool(physical["available"]),
        "physical_stage_height_m": measured,
        "absolute_error_m": None,
        "relative_error": None,
        "relative_error_percent": None,
        "robust_spread_m": None,
        "bootstrap_95pct_ci_m": [None, None],
        "ground_before_estimate_m": None,
        "ground_after_estimate_m": None,
        "before_after_consistency": None,
        "ground_plane": None,
        "tolerance_m": float(physical["tolerance_m"]),
        "tolerance_interpretation": physical.get(
            "tolerance_interpretation", "initial_screening_threshold_not_calibrated_accuracy"
        ),
        "reason": reason,
        "plateau_selection": selection,
    }


def height_validation(records, config, candidates=None):
    windows = config["plateau_windows"]
    required = ("ground_before", "stage_top", "ground_after")
    physical = config["physical_stage_height"]
    manual_complete = all(windows.get(name) is not None for name in required)
    selection = None
    if manual_complete:
        points = {name: selected_window_points(records, windows[name]) for name in required}
        selection = {
            "method": "manually_reviewed_windows",
            "measured_height_used_for_selection": False,
            "selection_completed_before_physical_comparison": True,
            "ambiguous": False,
            "ambiguity_reasons": [],
            "groups": {
                name: {
                    "candidate_ids": [], "time_windows_sec": [windows[name]],
                    "candidate_sample_counts": [int(points[name].shape[0])],
                    "total_samples": int(points[name].shape[0]),
                }
                for name in required
            },
        }
    elif config.get("plateau_selection", {}).get("enabled", False) and candidates is not None:
        selection = classify_plateau_candidates(
            candidates, config["time_window"]["ramp_start_sec"],
            config["time_window"]["ramp_end_sec"], config["plateau_selection"],
        )
        if selection["ambiguous"]:
            status = "HEIGHT_WARN" if physical["available"] else "HEIGHT_PENDING"
            result = incomplete_height_result(
                physical, status,
                "Candidate identity is ambiguous; no height estimate or physical comparison was forced.",
                selection,
            )
            result["manual_plateau_windows_complete"] = False
            return result
        points = {
            name: points_for_candidate_group(
                records, candidates, selection["groups"][name]["candidate_ids"]
            )
            for name in required
        }
    else:
        status = "HEIGHT_WARN" if physical["available"] else "HEIGHT_PENDING"
        result = incomplete_height_result(
            physical, status,
            "Manual windows are incomplete and automatic temporal selection is unavailable.",
        )
        result["manual_plateau_windows_complete"] = False
        return result
    minimum = int(config["height_estimation"]["minimum_samples_per_window"])
    if any(points[name].shape[0] < minimum for name in required):
        status = "HEIGHT_WARN" if physical["available"] else "HEIGHT_PENDING"
        result = incomplete_height_result(
            physical, status, f"At least {minimum} accepted samples are required in every plateau group.",
            selection,
        )
        result["sample_count"] = {name: int(points[name].shape[0]) for name in required}
        return result
    result = estimate_height_from_points(
        points["ground_before"], points["stage_top"], points["ground_after"],
        physical, config["height_estimation"]["bootstrap_samples"],
        config["height_estimation"]["bootstrap_seed"],
    )
    result["manual_plateau_windows_complete"] = manual_complete
    result["manual_plateau_windows"] = windows if manual_complete else None
    result["plateau_selection"] = selection
    result["reason"] = (
        "Physical comparison used a height-independent temporal partition of pre-detected candidates."
        if not manual_complete else "Physical comparison used manually reviewed windows."
    )
    result["fixed_offset_assumption"] = "A fixed LiDAR/base height offset cancels between flat plateaus."
    result["ramp_lever_arm_caveat"] = "Only stable candidate samples are fit because ramp attitude creates lever-arm effects."
    return result


def fingerprint(path, include_hash=False):
    path = Path(path).resolve()
    if path.is_dir():
        return {
            "path": str(path),
            "type": "directory",
            "files": {
                str(item.relative_to(path)): {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
                for item in sorted(path.rglob("*")) if item.is_file()
            },
        }
    entry = {"path": str(path), "type": "file", "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if include_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entry["sha256"] = digest.hexdigest()
    return entry


def validate_output_path(output, readonly_paths):
    output = Path(output).resolve()
    for source in readonly_paths:
        source = Path(source).resolve()
        protected = source if source.is_dir() else source.parent
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(f"Output directory {output} overlaps protected input {protected}")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Explicit field names are required for empty CSV {path}")
        fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(json_ready(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plot_omega(output, arrays):
    full = arrays["full_0_50s"]
    for index, axis_name in enumerate(AXES):
        figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        axis.plot(full["times"], full["gicp"][:, index], label=f"GICP body omega {axis_name}", linewidth=1.2)
        axis.plot(full["times"], full["imu"][:, index], label=f"bias-corrected IMU omega {axis_name}", linewidth=1.0, alpha=0.85)
        axis.set(xlabel="bag-relative time [s]", ylabel="angular velocity [rad/s]", title=f"IMU vs GICP angular velocity: {axis_name}")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.savefig(output / f"imu_vs_gicp_omega_{axis_name}.png", dpi=160)
        plt.close(figure)
    ramp = arrays["ramp_25_35s"]
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(ramp["times"], ramp["gicp"][:, 1], label="GICP body pitch rate (SO3 log y)", linewidth=1.4)
    axis.plot(ramp["times"], ramp["imu"][:, 1], label="bias-corrected IMU gyro y", linewidth=1.1)
    axis.set(xlabel="bag-relative time [s]", ylabel="angular velocity [rad/s]", title="Ramp pitch-rate consistency")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output / "ramp_imu_vs_gicp_pitch_rate.png", dpi=160)
    plt.close(figure)


def plot_orientation(output, rows):
    times = np.asarray([row["offset_sec"] for row in rows])
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(times, np.degrees([row["gicp_relative_pitch_rad"] for row in rows]), label="GICP relative pitch")
    axis.plot(times, np.degrees([row["imu_relative_pitch_rad"] for row in rows]), label="integrated gyro relative pitch")
    axis.set(xlabel="bag-relative time [s]", ylabel="relative pitch [deg]", title="Ramp relative pitch from common GICP anchor")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output / "ramp_relative_pitch.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(times, np.degrees([row["rotation_angle_error_rad"] for row in rows]))
    axis.set(xlabel="bag-relative time [s]", ylabel="SO(3) rotation error [deg]", title="Ramp integrated-gyro vs GICP relative orientation error")
    axis.grid(True, alpha=0.3)
    figure.savefig(output / "ramp_relative_orientation_error.png", dpi=160)
    plt.close(figure)


def plot_height(output, records, candidates, height):
    accepted = [record for record in records if record["accepted"]]
    times = np.asarray([record["offset"] for record in accepted])
    positions = np.asarray([record["position"] for record in accepted])
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(times, positions[:, 2], linewidth=1.2, label="accepted GICP z")
    group_colors = {
        "ground_before": "tab:blue", "stage_top": "tab:orange",
        "ground_after": "tab:green", "ambiguous_boundary_overlap": "tab:red",
    }
    used_labels = set()
    for candidate in candidates:
        group = candidate.get("temporal_group", "unclassified")
        label = group if group not in used_labels else None
        axis.axvspan(
            candidate["start_offset_sec"], candidate["end_offset_sec"],
            color=group_colors.get(group, "tab:gray"), alpha=0.17, label=label,
        )
        used_labels.add(group)
    axis.set(xlabel="bag-relative time [s]", ylabel="GICP z [m]", title="GICP z and rate/duration plateau candidates")
    axis.set_xlim(20.0, 40.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output / "gicp_z_with_plateau_candidates.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    if height.get("ground_plane"):
        plane = height["ground_plane"]
        residual = positions[:, 2] - (
            plane["a_dz_dx"] * positions[:, 0] + plane["b_dz_dy"] * positions[:, 1] + plane["c_m"]
        )
        axis.plot(times, residual, label="ground-plane residual")
        if height.get("measured_stage_height_m") is not None:
            axis.axhline(
                height["measured_stage_height_m"], color="black", linestyle="--",
                label="measured stage height",
            )
        selected_labels = set()
        for candidate in candidates:
            if not candidate.get("selected_for_height", False):
                continue
            selected = (times >= candidate["start_offset_sec"]) & (times <= candidate["end_offset_sec"])
            group = candidate["temporal_group"]
            axis.scatter(
                times[selected], residual[selected], s=18, color=group_colors[group],
                label=group if group not in selected_labels else None,
            )
            selected_labels.add(group)
        axis.set_ylabel("plane-corrected height [m]")
    else:
        axis.plot(times, positions[:, 2], label="raw accepted GICP z")
        axis.text(0.5, 0.9, "No manual plateau windows: ground-plane correction pending", transform=axis.transAxes, ha="center")
        axis.set_ylabel("GICP z [m]")
    axis.set(xlabel="bag-relative time [s]", title="Ground-plane-corrected stage height validation")
    axis.set_xlim(20.0, 40.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output / "ground_plane_corrected_height.png", dpi=160)
    plt.close(figure)


def gate_decision(axis_metrics, correlation_rows, lag_summary, orientation, config):
    metrics = {(row["window"], row["axis"]): row for row in axis_metrics}
    correlation = {
        (row["window"], row["gicp_axis"], row["imu_axis"]): row for row in correlation_rows
    }
    warnings = []
    failures = []
    full_count = metrics[("full_0_50s", "x")]["usable_samples"]
    ramp_count = metrics[("ramp_25_35s", "x")]["usable_samples"]
    if full_count < config["minimum_usable_pairs_full"]:
        failures.append(f"full usable pairs {full_count} < {config['minimum_usable_pairs_full']}")
    if ramp_count < config["minimum_usable_pairs_ramp"]:
        failures.append(f"ramp usable pairs {ramp_count} < {config['minimum_usable_pairs_ramp']}")
    mapping = {}
    for axis in AXES:
        expected = metrics[("full_0_50s", axis)]
        best_row = next(
            row for row in correlation_rows
            if row["window"] == "full_0_50s" and row["gicp_axis"] == axis and row["best_absolute_mapping"]
        )
        excited = expected["gicp_std_radps"] >= config["minimum_axis_excitation_std_radps"]
        mapping[axis] = {
            "expected_imu_axis": axis,
            "best_absolute_imu_axis": best_row["imu_axis"],
            "expected_pearson": expected["pearson"],
            "expected_sign_agreement": expected["sign_agreement"],
            "sufficiently_excited": excited,
        }
        if expected["pearson"] < 0.0 and excited:
            failures.append(f"{axis} expected-axis correlation has opposite sign ({expected['pearson']:.3f})")
        elif excited and expected["pearson"] < config["minimum_expected_axis_pearson"]:
            warnings.append(f"{axis} expected-axis Pearson {expected['pearson']:.3f} is below initial threshold")
        if excited and expected["sign_agreement"] < config["minimum_expected_axis_sign_agreement"]:
            warnings.append(f"{axis} sign agreement {expected['sign_agreement']:.3f} is below initial threshold")
        if best_row["imu_axis"] != axis:
            warnings.append(f"GICP {axis} best absolute correlation is IMU {best_row['imu_axis']}, not expected {axis}")
        lag = lag_summary["full_0_50s"][axis]["best_lag_sec"]
        if lag is None or abs(lag) > config["maximum_reasonable_best_lag_sec"]:
            warnings.append(f"{axis} best lag {lag} s is outside the initial reasonable range")
    ramp_pitch = metrics[("ramp_25_35s", "y")]
    if ramp_pitch["pearson"] < 0.0:
        failures.append("ramp pitch-rate correlation has opposite sign")
    elif ramp_pitch["pearson"] < config["minimum_ramp_pitch_pearson"]:
        warnings.append(f"ramp pitch Pearson {ramp_pitch['pearson']:.3f} is below initial threshold")
    if orientation["rotation_error_p95_deg"] > config["maximum_orientation_error_fail_deg"]:
        failures.append("ramp relative orientation error exceeds failure threshold")
    elif orientation["rotation_error_p95_deg"] > config["maximum_orientation_error_p95_deg"]:
        warnings.append("ramp relative orientation p95 error exceeds initial pass threshold")
    status = "GYRO_FAIL" if failures else ("GYRO_WARN" if warnings else "GYRO_PASS")
    return status, {"failures": failures, "warnings": warnings, "axis_mapping": mapping}


def report_text(summary, axis_metrics, candidates, height):
    metric = {(row["window"], row["axis"]): row for row in axis_metrics}
    lag = summary["time_lag"]
    bias = summary["gyro_bias"]
    orientation = summary["ramp_orientation"]
    candidate_lines = "\n".join(
        f"- Candidate {row['candidate_id']}: {row['start_offset_sec']:.3f}-{row['end_offset_sec']:.3f} s, "
        f"duration {row['duration_sec']:.3f} s, n={row['sample_count']}, median z={row['median_z_m']:.4f} m, "
        f"median pitch={row['median_pitch_deg']:.3f} deg, z MAD={row['z_mad_m']:.4f} m, "
        f"temporal group=`{row.get('temporal_group', 'unclassified')}`"
        for row in candidates
    ) or "- No candidate met the configured stability and duration thresholds."
    axis_lines = []
    for window in ("full_0_50s", "ramp_25_35s"):
        for axis in AXES:
            row = metric[(window, axis)]
            axis_lines.append(
                f"| {window} | {axis} ({row['motion_name']}) | {row['usable_samples']} | "
                f"{row['pearson']:.4f} | {row['spearman']:.4f} | {row['rmse_radps']:.4f} | "
                f"{row['mae_radps']:.4f} | {row['sign_agreement']:.4f} |"
            )
    lag_lines = []
    for window in ("full_0_50s", "ramp_25_35s"):
        for axis in AXES:
            row = lag[window][axis]
            lag_lines.append(
                f"| {window} | {axis} | {row['zero_lag_pearson']:.4f} | "
                f"{row['best_lag_sec']:+.3f} | {row['best_lag_pearson']:.4f} |"
            )
    mapping_lines = []
    for axis in AXES:
        row = summary["gyro_diagnostics"]["axis_mapping"][axis]
        sign = "same" if row["expected_pearson"] >= 0.0 else "opposite"
        mapping_lines.append(
            f"| {axis} ({EXPECTED_MOTION_NAMES[axis]}) | {row['expected_imu_axis']} | "
            f"{row['best_absolute_imu_axis']} | {row['expected_pearson']:.4f} | {sign} |"
        )
    height_estimate = (
        "not estimated because plateau identity is ambiguous"
        if height.get("estimated_stage_height_m") is None
        else f"{height['estimated_stage_height_m']:.6f} m"
    )
    selection = height.get("plateau_selection") or {}
    selection_groups = selection.get("groups", {})
    selection_lines = []
    for name in ("ground_before", "stage_top", "ground_after"):
        group = selection_groups.get(name, {})
        windows = group.get("time_windows_sec", [])
        selection_lines.append(
            f"| {name} | {group.get('candidate_ids', [])} | {group.get('candidate_sample_counts', [])} | "
            f"{windows} | {group.get('total_samples', 0)} |"
        )
    measured = height.get("measured_stage_height_m")
    absolute_error = height.get("absolute_error_m")
    relative_error_percent = height.get("relative_error_percent")
    robust_spread = height.get("robust_spread_m")
    confidence = height.get("bootstrap_95pct_ci_m")
    before_estimate = height.get("ground_before_estimate_m")
    after_estimate = height.get("ground_after_estimate_m")
    consistency = height.get("before_after_consistency") or {}
    plane = height.get("ground_plane") or {}
    optional = lambda value, digits=6: "not available" if value is None else f"{value:.{digits}f}"
    return f"""# IMU–GICP physical consistency Gate

## Decision

- Gyro consistency: **{summary['gyro_status']}**
- Height consistency: **{summary['height_status']}**
- Overall: **{summary['overall_status']}**

This Gate is a read-only validation of the existing production planar-EKF/full-6DoF-small_gicp
result. It does not change or rerun localization, GICP, EKF, voxel, rejection, or timestamp
parameters. All analyzed timestamps are within bag-relative 0-50 s.

## Observed facts

- Bag messages used: {summary['input_counts']['imu']} IMU and {summary['input_counts']['odom']} odometry.
- Identity IMU orientation messages: {summary['input_counts']['identity_imu_orientation']} / {summary['input_counts']['imu']}; orientation was not used.
- GICP adjacent-pair candidates/usable after IMU coverage: {summary['pairing']['valid_gicp_pairs']} / {summary['pairing']['usable_imu_gicp_pairs']}.
- Excluded GICP pairs: {summary['pairing']['excluded']}.
- Robust stationary gyro median bias [x,y,z]: {bias['median_radps'][0]:+.8f}, {bias['median_radps'][1]:+.8f}, {bias['median_radps'][2]:+.8f} rad/s from {bias['sample_count']} samples.
- Stationary selection maxima, odom linear/angular and gyro norm: {bias['selected_odom_linear_speed_max_mps']:.6f} m/s / {bias['selected_odom_angular_speed_max_radps']:.6f} rad/s / {bias['selected_gyro_norm_max_radps']:.6f} rad/s.
- Ramp integration anchor: accepted GICP orientation at {orientation['anchor_offset_sec']:.6f} s. This is a common relative anchor, not an absolute IMU orientation measurement.
- Ramp orientation error mean/p95/max: {orientation['rotation_error_mean_deg']:.4f} / {orientation['rotation_error_p95_deg']:.4f} / {orientation['rotation_error_max_deg']:.4f} deg.
- Ramp net GICP relative RPY: {orientation['gicp_net_relative_rpy_deg']} deg.
- Ramp net integrated-gyro relative RPY: {orientation['imu_net_relative_rpy_deg']} deg.
- Ramp net GICP/integrated-gyro rotation angle: {orientation['gicp_net_relative_rotation_angle_deg']:.4f} / {orientation['imu_net_relative_rotation_angle_deg']:.4f} deg.
- Input fingerprints were unchanged: {summary['input_integrity']['unchanged']}.

| window | axis | usable | Pearson | Spearman | RMSE [rad/s] | MAE [rad/s] | sign agreement |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(axis_lines)}

The primary GICP rate is `Log(R_k^T R_(k+1))/dt`, expressed in the LiDAR/body frame. Euler
angle differences are not used for the primary gyro comparison.

## Axis/sign diagnosis

| GICP body axis | expected IMU axis | best absolute-correlation IMU axis | expected-axis Pearson | sign |
|---|---|---|---:|---|
{chr(10).join(mapping_lines)}

The expected x/y/z mapping remains the best mapping on all axes, and none has an inverted
correlation sign. The x/roll evidence is weak enough to remain a warning rather than being
reported as a strong validation.

Ramp pitch timing/sign: zero-lag Pearson {metric[('ramp_25_35s', 'y')]['pearson']:.4f}, sign
agreement {metric[('ramp_25_35s', 'y')]['sign_agreement']:.4f}, GICP peak at
{metric[('ramp_25_35s', 'y')]['gicp_peak_offset_sec']:.6f} s, and IMU peak at
{metric[('ramp_25_35s', 'y')]['imu_peak_offset_sec']:.6f} s.

## Timestamp-lag diagnosis

Lag `L` compares GICP(t) with IMU(t+L); positive lag queries a later IMU timestamp.
Best lag maximizes absolute correlation inside contiguous accepted-pair segments. It is reported
only; no timestamp offset is applied to production data.

| window | axis | zero-lag Pearson | best lag [s] | best-lag Pearson |
|---|---|---:|---:|---:|
{chr(10).join(lag_lines)}

## Stable-plateau candidates, 20-40 s

{candidate_lines}

Candidate creation uses only smoothed absolute z rate, absolute pitch rate, duration, accepted-pose
continuity, and timestamp-gap limits. The measured 0.150 m is not passed to detection or temporal
partitioning. After detection, candidates are partitioned solely by their time relation to the
predeclared 25-35 s ramp window.

| temporal group | candidate IDs | per-candidate samples | time windows [s] | total samples |
|---|---|---|---|---:|
{chr(10).join(selection_lines)}

- Selection method: `{selection.get('method')}`
- Selection ambiguous: {selection.get('ambiguous')}
- Selection ambiguity reasons: {selection.get('ambiguity_reasons')}
- Measured height used for selection: {selection.get('measured_height_used_for_selection')}
- Surface-identity basis: `{selection.get('surface_identity_basis')}`; external label confirmation: {selection.get('external_surface_label_confirmation')}

## Height validation

- Status: **{height['status']}**
- Measured stage height: {optional(measured)} m
- Measurement: `{height.get('measurement_method')}`; reference: `{height.get('measurement_reference')}`
- Estimated stage height: {height_estimate}
- Raw absolute error: {optional(absolute_error)} m
- Raw relative error: {optional(relative_error_percent, 3)} %
- Stage residual robust spread: {optional(robust_spread)} m
- Bootstrap 95% confidence interval: {confidence} m
- Ground-before-referenced estimate: {optional(before_estimate)} m
- Ground-after-referenced estimate: {optional(after_estimate)} m
- Before/after consistency: {consistency.get('status')} (difference {optional(consistency.get('difference_m'))} m; initial limit {optional(consistency.get('tolerance_m'))} m)
- Ground plane `z = a*x + b*y + c`: a={optional(plane.get('a_dz_dx'))}, b={optional(plane.get('b_dz_dy'))}, c={optional(plane.get('c_m'))} m
- Ground-plane slope: {optional(plane.get('slope_magnitude'))} m/m ({optional(plane.get('slope_angle_deg'), 3)} deg); warning={plane.get('slope_warning')}
- Physical stage height available: {height['physical_stage_height_available']}
- Screening tolerance: {height['tolerance_m']:.3f} m
- Tolerance interpretation: `{height['tolerance_interpretation']}`
- Raw 25-35 s z max-min: {height['raw_ramp_z_range_m']:.6f} m; used as height estimate: {height['raw_ramp_z_range_used_for_estimate']}
- Reason: {height.get('reason', 'Manual windows were evaluated.')}

A fixed LiDAR/base vertical offset cancels in relative flat-plateau height. Ramp attitude can
create a lever-arm effect, so ramp samples are not treated as plateau height evidence. The raw
z range is never used as a stage-height estimate. The 0.050 m tolerance is an initial physical
consistency screening threshold chosen with the 0.20 m map/scan voxel resolution and manual
measurement in mind; it is not a calibrated localization-accuracy threshold. Raw errors are
reported independently of that threshold.

The reported ground-plane slope warning is an observation from the selected GICP poses. Map tilt
and localization-z inconsistency are possible explanations, but this Gate does not distinguish
between them.

## Interpretation (not directly observed)

The axis-mapping and lag checks are diagnostics, not automatic extrinsic or clock calibration.
Any unexpected best axis, sign, or lag remains a reported warning; this tool does not silently
swap axes, invert signs, or apply time offsets. Accelerometer double integration is not used.

Initial-threshold warnings: {summary['gyro_diagnostics']['warnings']}

Initial-threshold failures: {summary['gyro_diagnostics']['failures']}
"""


def run(config_path, output_directory):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    inputs = config["inputs"]
    output = Path(output_directory).resolve()
    readonly_paths = [
        inputs["bag"], Path(inputs["localization_csv"]).parent,
        inputs["map_ply"], inputs["glim_dump"],
        *inputs.get("preserved_result_directories", []),
    ]
    validate_output_path(output, readonly_paths)
    before = {
        "bag": fingerprint(inputs["bag"]),
        "localization_csv": fingerprint(inputs["localization_csv"], include_hash=True),
        "map_ply": fingerprint(inputs["map_ply"]),
        "glim_dump": fingerprint(inputs["glim_dump"]),
    }
    for index, path in enumerate(inputs.get("preserved_result_directories", [])):
        before[f"preserved_result_{index}"] = fingerprint(path)
    output.mkdir(parents=True, exist_ok=True)
    window = config["time_window"]
    records = read_localization_csv(
        inputs["localization_csv"], window["origin_timestamp"], window["start_sec"], window["end_sec"]
    )
    sensor = bag_sensor_data(
        inputs["bag"], window["origin_timestamp"], window["start_sec"], window["end_sec"]
    )
    bias, _ = select_stationary_bias(sensor, window["origin_timestamp"], config["gyro_bias"])
    lidar_R_imu = quaternion_to_rotation(config["extrinsic"]["rotation_xyzw"])
    pairs, excluded = build_gicp_pairs(records, config["pairing"]["max_pair_dt_sec"])
    usable_pairs, corrected_gyro = attach_imu_averages(
        pairs, sensor, bias["median_radps"], config["imu_interval"], lidar_R_imu
    )
    axis_metrics, arrays = compute_axis_metrics(
        usable_pairs, window["ramp_start_sec"], window["ramp_end_sec"],
        config["gyro_gate"]["sign_deadband_radps"],
    )
    correlation_rows = compute_axis_correlation_matrices(arrays)
    lag_rows, lag_summary = compute_lag_rows(arrays, config["lag_search"])
    orientation_rows, orientation_summary = orientation_comparison(
        records, sensor["imu_times"], corrected_gyro, lidar_R_imu,
        window["ramp_start_sec"], window["ramp_end_sec"], config["imu_interval"]["max_sample_gap_sec"],
    )
    candidates = detect_plateaus(records, config["plateau_detection"], config["pairing"]["max_pair_dt_sec"])
    height = height_validation(records, config, candidates)
    ramp_z = np.asarray([
        record["position"][2] for record in records
        if record["accepted"] and window["ramp_start_sec"] <= record["offset"] <= window["ramp_end_sec"]
    ])
    height["raw_ramp_z_range_m"] = float(np.ptp(ramp_z))
    height["raw_ramp_z_range_used_for_estimate"] = False
    gyro_status, gyro_diagnostics = gate_decision(
        axis_metrics, correlation_rows, lag_summary, orientation_summary, config["gyro_gate"]
    )
    height_status = height["status"]
    if gyro_status == "GYRO_FAIL" or height_status == "HEIGHT_FAIL":
        overall = "OVERALL_FAIL"
    elif gyro_status == "GYRO_WARN" or height_status == "HEIGHT_WARN":
        overall = "OVERALL_WARN_WITH_PENDING_HEIGHT" if height_status == "HEIGHT_PENDING" else "OVERALL_WARN"
    elif height_status == "HEIGHT_PENDING":
        overall = "OVERALL_PASS_WITH_PENDING_HEIGHT"
    else:
        overall = "OVERALL_PASS"

    after = {
        "bag": fingerprint(inputs["bag"]),
        "localization_csv": fingerprint(inputs["localization_csv"], include_hash=True),
        "map_ply": fingerprint(inputs["map_ply"]),
        "glim_dump": fingerprint(inputs["glim_dump"]),
    }
    for index, path in enumerate(inputs.get("preserved_result_directories", [])):
        after[f"preserved_result_{index}"] = fingerprint(path)
    integrity = {"unchanged": before == after, "before": before, "after": after}
    if not integrity["unchanged"]:
        raise RuntimeError("A protected input changed during read-only analysis")
    summary = {
        "gate_name": "IMU-GICP physical consistency Gate",
        "gyro_status": gyro_status,
        "height_status": height_status,
        "overall_status": overall,
        "scope": {
            "bag_relative_window_sec": [window["start_sec"], window["end_sec"]],
            "ramp_window_sec": [window["ramp_start_sec"], window["ramp_end_sec"]],
            "localization_parameters_changed": False,
            "imu_orientation_used": False,
            "accelerometer_used": False,
            "axis_or_timestamp_correction_applied": False,
        },
        "input_counts": {
            "localization_rows": len(records),
            "accepted_localization_rows": sum(record["accepted"] for record in records),
            "imu": int(sensor["imu_times"].size),
            "odom": int(sensor["odom_times"].size),
            "identity_imu_orientation": sensor["imu_orientation_identity_count"],
        },
        "input_time_bounds_sec": {
            "localization_first": records[0]["offset"],
            "localization_last": records[-1]["offset"],
            "imu_first": float(sensor["imu_times"][0] - window["origin_timestamp"]),
            "imu_last": float(sensor["imu_times"][-1] - window["origin_timestamp"]),
            "odom_first": float(sensor["odom_times"][0] - window["origin_timestamp"]),
            "odom_last": float(sensor["odom_times"][-1] - window["origin_timestamp"]),
        },
        "gyro_bias": bias,
        "pairing": {
            "adjacent_possible_pairs": len(records) - 1,
            "valid_gicp_pairs": len(pairs),
            "usable_imu_gicp_pairs": len(usable_pairs),
            "excluded": excluded,
            "imu_coverage_excluded": len(pairs) - len(usable_pairs),
            "max_pair_dt_sec": config["pairing"]["max_pair_dt_sec"],
        },
        "time_lag": lag_summary,
        "ramp_orientation": orientation_summary,
        "plateau_candidate_count": len(candidates),
        "height_validation": height,
        "gyro_diagnostics": gyro_diagnostics,
        "initial_thresholds": config["gyro_gate"],
        "input_integrity": integrity,
    }

    write_json(output / "gyro_bias.json", bias)
    write_csv(output / "imu_gicp_axis_metrics.csv", axis_metrics)
    write_csv(output / "axis_correlation_matrix.csv", correlation_rows)
    write_csv(output / "time_lag_metrics.csv", lag_rows)
    write_csv(output / "orientation_comparison.csv", orientation_rows)
    write_csv(
        output / "candidate_plateaus.csv", candidates,
        ["candidate_id", "start_offset_sec", "end_offset_sec", "duration_sec", "sample_count",
         "median_x_m", "median_y_m", "median_z_m", "median_pitch_rad", "median_pitch_deg", "z_mad_m",
         "temporal_group", "selected_for_height"],
    )
    write_csv(output / "imu_gicp_pairs.csv", [
        {
            "start_offset_sec": pair["start_offset"], "end_offset_sec": pair["end_offset"],
            "mid_offset_sec": pair["mid_offset"], "dt_sec": pair["dt"], "segment": pair["segment"],
            **{f"gicp_omega_{axis}_radps": pair["omega_gicp"][index] for index, axis in enumerate(AXES)},
            **{f"imu_omega_{axis}_radps": pair["omega_imu"][index] for index, axis in enumerate(AXES)},
        }
        for pair in usable_pairs
    ])
    write_json(output / "height_validation.json", height)
    write_json(output / "physical_validation_summary.json", summary)
    (output / "physical_validation_report.md").write_text(
        report_text(summary, axis_metrics, candidates, height), encoding="utf-8"
    )
    plot_omega(output, arrays)
    plot_orientation(output, orientation_rows)
    plot_height(output, records, candidates, height)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    summary = run(args.config, args.output_directory)
    print(json.dumps({
        "gyro_status": summary["gyro_status"],
        "height_status": summary["height_status"],
        "overall_status": summary["overall_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
