#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import statistics
import sys
import time
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_prefix
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2


EXPECTED_MAP_SHA256 = "b3a208bf44c7f71db848b676641192f272eabf7fb07af153f26d42f6ca847240"
DEFAULT_MAP = Path(
    "/home/a/Desktop/shihoon/glim_real/20260819_flat/results/flat_bag_C_imu_on.ply"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_seconds(message):
    return float(message.header.stamp.sec) + 1.0e-9 * float(message.header.stamp.nanosec)


def package_available(name):
    try:
        return bool(get_package_prefix(name))
    except PackageNotFoundError:
        return False


def percentile(values, percent):
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def unwrap(values):
    if not values:
        return []
    output = [values[0]]
    for value in values[1:]:
        output.append(output[-1] + math.remainder(value - output[-1], 2.0 * math.pi))
    return output


def analyze_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        print(f"FAIL no localization records in {path}", file=sys.stderr)
        return 1
    accepted = [row for row in rows if row["accepted"] == "1"]
    registrations = [
        row
        for row in rows
        if row["prediction_available"] == "1" and int(row["finite_points"]) >= 10
    ]
    if not registrations:
        print("FAIL no registration attempts found", file=sys.stderr)
        return 1

    def values(records, key):
        return [float(row[key]) for row in records]

    runtimes = values(registrations, "runtime_ms")
    corrections = values(registrations, "correction_translation_m")
    yaw_corrections = [abs(value) for value in values(registrations, "correction_yaw_rad")]
    print(f"processed={len(rows)} accepted={len(accepted)} rejected={len(rows) - len(accepted)}")
    print(f"accepted_pose_rate={len(accepted) / len(rows):.6f}")
    print(
        "gicp_runtime_ms "
        f"mean={statistics.fmean(runtimes):.3f} p95={percentile(runtimes, 0.95):.3f} "
        f"p99={percentile(runtimes, 0.99):.3f}"
    )
    print(
        "registration_correction_m "
        f"mean={statistics.fmean(corrections):.6f} "
        f"p95={percentile(corrections, 0.95):.6f} "
        f"p99={percentile(corrections, 0.99):.6f}"
    )
    print(
        "registration_correction_abs_yaw_rad "
        f"mean={statistics.fmean(yaw_corrections):.6f} "
        f"p95={percentile(yaw_corrections, 0.95):.6f} "
        f"p99={percentile(yaw_corrections, 0.99):.6f}"
    )
    if accepted:
        for key in ("gicp_x", "gicp_y"):
            samples = values(accepted, key)
            print(
                f"accepted_{key} range={max(samples) - min(samples):.6f} "
                f"std={statistics.pstdev(samples):.6f}"
            )
        yaw = unwrap(values(accepted, "gicp_yaw"))
        print(
            f"accepted_gicp_yaw_rad range={max(yaw) - min(yaw):.6f} "
            f"std={statistics.pstdev(yaw):.6f}"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Read-only Live Real Localization V1 preflight")
    parser.add_argument("--map-path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--expected-lidar-frame", default="velodyne")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--max-stamp-age-sec", type=float, default=5.0)
    parser.add_argument(
        "--analyze-csv",
        type=Path,
        help="analyze a completed stationary localization.csv instead of probing live topics",
    )
    args = parser.parse_args()
    if args.analyze_csv is not None:
        return analyze_csv(args.analyze_csv)

    failures = []
    if not args.map_path.is_file():
        failures.append(f"map does not exist: {args.map_path}")
    else:
        actual_hash = sha256(args.map_path)
        if actual_hash != EXPECTED_MAP_SHA256:
            failures.append(f"map SHA256 mismatch: {actual_hash}")
        else:
            print(f"PASS map SHA256 {actual_hash}")

    for package in ("bunker_offline_localization", "robot_localization"):
        if package_available(package):
            print(f"PASS package available: {package}")
        else:
            failures.append(f"package unavailable in sourced environment: {package}")

    rclpy.init()
    node = rclpy.create_node("live_real_localization_v1_preflight")
    qos = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    messages = {"/odom": [], "/imu/data": [], "/velodyne_points": []}

    def callback(topic):
        def receive(message):
            if len(messages[topic]) < 2:
                messages[topic].append(message)

        return receive

    subscriptions = [
        node.create_subscription(Odometry, "/odom", callback("/odom"), qos),
        node.create_subscription(Imu, "/imu/data", callback("/imu/data"), qos),
        node.create_subscription(
            PointCloud2, "/velodyne_points", callback("/velodyne_points"), qos
        ),
    ]
    del subscriptions

    deadline = time.monotonic() + args.timeout_sec
    while time.monotonic() < deadline and any(len(items) < 2 for items in messages.values()):
        rclpy.spin_once(node, timeout_sec=0.1)

    now = node.get_clock().now().nanoseconds * 1.0e-9
    for topic, items in messages.items():
        publisher_count = node.count_publishers(topic)
        if publisher_count < 1:
            failures.append(f"no publisher discovered on {topic}")
            continue
        if len(items) < 2:
            failures.append(f"fewer than two messages received on {topic}")
            continue
        stamps = [stamp_seconds(item) for item in items]
        if stamps[0] <= 0.0 or stamps[1] <= stamps[0]:
            failures.append(f"timestamps are not positive and increasing on {topic}: {stamps}")
            continue
        age = abs(now - stamps[-1])
        if age > args.max_stamp_age_sec:
            failures.append(f"latest {topic} stamp differs from live clock by {age:.3f} s")
            continue
        print(
            f"PASS {topic}: publishers={publisher_count} stamps increase "
            f"({stamps[0]:.9f} -> {stamps[1]:.9f})"
        )

    if messages["/velodyne_points"]:
        frame = messages["/velodyne_points"][-1].header.frame_id
        if frame != args.expected_lidar_frame:
            failures.append(
                f"/velodyne_points frame_id is {frame!r}, expected {args.expected_lidar_frame!r}"
            )
        else:
            print(f"PASS /velodyne_points frame_id={frame}")

    cmd_vel_publishers = node.get_publishers_info_by_topic("/cmd_vel")
    if cmd_vel_publishers:
        names = ", ".join(
            sorted({f"{info.node_namespace}/{info.node_name}" for info in cmd_vel_publishers})
        )
        failures.append(f"active /cmd_vel publisher(s) detected: {names}")
    else:
        print("PASS no active /cmd_vel publisher")

    node.destroy_node()
    rclpy.shutdown()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("PASS Live Real Localization V1 preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
