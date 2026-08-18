#!/usr/bin/env python3
"""Receive and validate every required GICP diagnostic visualization topic."""

import argparse
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray


EXPECTED = {
    "/map_cloud": (PointCloud2, "sensor_msgs/msg/PointCloud2", "map"),
    "/raw_scan": (PointCloud2, "sensor_msgs/msg/PointCloud2", "velodyne"),
    "/registered_scan": (PointCloud2, "sensor_msgs/msg/PointCloud2", "map"),
    "/gicp_pose": (PoseStamped, "geometry_msgs/msg/PoseStamped", "map"),
    "/gicp_path": (PathMessage, "nav_msgs/msg/Path", "map"),
    "/prediction_pose": (PoseStamped, "geometry_msgs/msg/PoseStamped", "map"),
    "/gicp_correspondences": (
        MarkerArray,
        "visualization_msgs/msg/MarkerArray",
        "map",
    ),
}


def message_frame(message):
    if isinstance(message, MarkerArray):
        frames = [marker.header.frame_id for marker in message.markers if marker.header.frame_id]
        return frames[0] if frames else ""
    return message.header.frame_id


class TopicVerifier(Node):
    def __init__(self, timeout_sec):
        super().__init__("gicp_rviz_topic_verifier")
        self.received = {}
        self.started = time.monotonic()
        self.timeout_sec = timeout_sec
        transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        live = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for topic, (message_type, type_name, expected_frame) in EXPECTED.items():
            qos = transient if topic == "/map_cloud" else live
            self.create_subscription(
                message_type,
                topic,
                lambda message, name=topic, declared=type_name, frame=expected_frame: self._receive(
                    name, declared, frame, message
                ),
                qos,
            )

    def _receive(self, topic, type_name, expected_frame, message):
        frame = message_frame(message)
        self.received[topic] = {
            "type": type_name,
            "frame_id": frame,
            "expected_frame_id": expected_frame,
            "frame_valid": frame == expected_frame,
        }

    def complete(self):
        return len(self.received) == len(EXPECTED) or (
            time.monotonic() - self.started >= self.timeout_sec
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = TopicVerifier(args.timeout_sec)
    while rclpy.ok() and not node.complete():
        rclpy.spin_once(node, timeout_sec=0.2)
    missing = sorted(set(EXPECTED) - set(node.received))
    frames_valid = all(item["frame_valid"] for item in node.received.values())
    result = {
        "all_required_topics_received": not missing,
        "all_frame_ids_valid": frames_valid,
        "missing_topics": missing,
        "topics": node.received,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if not missing and frames_valid else 1)


if __name__ == "__main__":
    main()
