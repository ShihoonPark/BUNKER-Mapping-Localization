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
DYNAMIC_TOPICS = set(EXPECTED) - {"/map_cloud"}
EXPECTED_QOS = {
    "/map_cloud": (ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL),
    **{
        topic: (ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE)
        for topic in DYNAMIC_TOPICS
    },
}


def message_frame(message):
    if isinstance(message, MarkerArray):
        frames = [marker.header.frame_id for marker in message.markers if marker.header.frame_id]
        return frames[0] if frames else ""
    return message.header.frame_id


def policy_name(policy):
    return getattr(policy, "name", str(policy))


class TopicVerifier(Node):
    def __init__(self, timeout_sec, minimum_dynamic_messages):
        super().__init__("gicp_rviz_topic_verifier")
        self.received = {}
        self.counts = {topic: 0 for topic in EXPECTED}
        self.started = time.monotonic()
        self.timeout_sec = timeout_sec
        self.minimum_dynamic_messages = minimum_dynamic_messages
        self.publisher_profiles_seen = {topic: [] for topic in EXPECTED}
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
        self.counts[topic] += 1
        self.received[topic] = {
            "type": type_name,
            "frame_id": frame,
            "expected_frame_id": expected_frame,
            "frame_valid": frame == expected_frame,
        }

    def complete(self):
        enough_messages = all(
            self.counts[topic] >= (
                self.minimum_dynamic_messages if topic in DYNAMIC_TOPICS else 1
            )
            for topic in EXPECTED
        )
        return enough_messages or time.monotonic() - self.started >= self.timeout_sec

    def sample_publisher_qos(self):
        for topic in EXPECTED:
            profiles = self.get_publishers_info_by_topic(topic)
            if profiles:
                self.publisher_profiles_seen[topic] = profiles

    def publisher_qos(self):
        result = {}
        for topic, (expected_reliability, expected_durability) in EXPECTED_QOS.items():
            profiles = []
            for endpoint in self.publisher_profiles_seen[topic]:
                qos = endpoint.qos_profile
                profiles.append({
                    "node_name": endpoint.node_name,
                    "reliability": policy_name(qos.reliability),
                    "durability": policy_name(qos.durability),
                    "compatible": (
                        qos.reliability == expected_reliability
                        and qos.durability == expected_durability
                    ),
                })
            result[topic] = {
                "expected_reliability": policy_name(expected_reliability),
                "expected_durability": policy_name(expected_durability),
                "offered_profiles": profiles,
                "valid": bool(profiles) and all(item["compatible"] for item in profiles),
            }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--minimum-dynamic-messages", type=int, default=1)
    args = parser.parse_args()
    if args.minimum_dynamic_messages < 1:
        parser.error("--minimum-dynamic-messages must be at least 1")
    rclpy.init()
    node = TopicVerifier(args.timeout_sec, args.minimum_dynamic_messages)
    while rclpy.ok() and not node.complete():
        rclpy.spin_once(node, timeout_sec=0.2)
        node.sample_publisher_qos()
    node.sample_publisher_qos()
    missing = sorted(
        topic for topic in EXPECTED
        if node.counts[topic] < (
            args.minimum_dynamic_messages if topic in DYNAMIC_TOPICS else 1
        )
    )
    frames_valid = all(item["frame_valid"] for item in node.received.values())
    publisher_qos = node.publisher_qos()
    qos_valid = all(item["valid"] for item in publisher_qos.values())
    result = {
        "all_required_topics_received": not missing,
        "all_frame_ids_valid": frames_valid,
        "all_publisher_qos_valid": qos_valid,
        "minimum_dynamic_messages": args.minimum_dynamic_messages,
        "message_counts": node.counts,
        "missing_topics": missing,
        "publisher_qos": publisher_qos,
        "topics": node.received,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if not missing and frames_valid and qos_valid else 1)


if __name__ == "__main__":
    main()
