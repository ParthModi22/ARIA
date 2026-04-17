"""ROS 2 node that retargets MediaPipe pose landmarks into OP3 joint commands."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


EXPECTED_FLOAT_COUNT = 33 * 4
LANDMARK_COUNT = 33
VALUES_PER_LANDMARK = 4


@dataclass(frozen=True)
class JointConfig:
    """Joint configuration loaded from YAML."""

    name: str
    mediapipe_landmarks: tuple[int, int, int]
    minimum: float
    maximum: float
    invert: bool
    offset: float


class RetargetingNode(Node):
    """Convert MediaPipe world landmarks into clamped joint positions."""

    def __init__(self) -> None:
        super().__init__("retargeting_node")

        self.joint_configs = self._load_joint_configs()
        self.publisher = self.create_publisher(JointState, "/op3/joint_commands", 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            self.landmarks_callback,
            10,
        )

        self._last_timing_log = time.monotonic()
        self.get_logger().info(
            f"retargeting_node started with {len(self.joint_configs)} configured joints"
        )

    def _load_joint_configs(self) -> list[JointConfig]:
        config_path = Path(__file__).with_name("op3_joint_limits.yaml")
        with config_path.open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream) or {}

        joint_limits = raw_config.get("joint_limits")
        if not isinstance(joint_limits, dict) or not joint_limits:
            raise ValueError("op3_joint_limits.yaml must define a non-empty joint_limits map")

        configs: list[JointConfig] = []
        for joint_name, spec in joint_limits.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Joint '{joint_name}' config must be a mapping")

            indices = spec.get("mediapipe_landmarks")
            if not isinstance(indices, list) or len(indices) != 3:
                raise ValueError(
                    f"Joint '{joint_name}' must define mediapipe_landmarks as 3 indices"
                )

            configs.append(
                JointConfig(
                    name=str(joint_name),
                    mediapipe_landmarks=tuple(int(index) for index in indices),
                    minimum=float(spec["min"]),
                    maximum=float(spec["max"]),
                    invert=bool(spec.get("invert", False)),
                    offset=float(spec.get("offset", 0.0)),
                )
            )

        return configs

    def landmarks_callback(self, msg: Float32MultiArray) -> None:
        start_time = time.perf_counter()

        if len(msg.data) != EXPECTED_FLOAT_COUNT:
            self.get_logger().warning(
                f"Expected {EXPECTED_FLOAT_COUNT} floats, received {len(msg.data)}"
            )
            return

        landmarks = self._unpack_landmarks(msg.data)
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [config.name for config in self.joint_configs]
        joint_state.position = [
            self._compute_joint_position(config, landmarks) for config in self.joint_configs
        ]
        self.publisher.publish(joint_state)

        compute_time_ms = (time.perf_counter() - start_time) * 1000.0
        now = time.monotonic()
        if now - self._last_timing_log >= 5.0:
            self.get_logger().info(f"Retargeting compute time: {compute_time_ms:.2f} ms")
            self._last_timing_log = now

    def _unpack_landmarks(self, flat_data: list[float]) -> list[np.ndarray]:
        points: list[np.ndarray] = []

        for index in range(LANDMARK_COUNT):
            base = index * VALUES_PER_LANDMARK
            mp_x = float(flat_data[base])
            mp_y = float(flat_data[base + 1])
            mp_z = float(flat_data[base + 2])
            _visibility = float(flat_data[base + 3])

            ros_point = np.array(
                [-mp_z, -mp_x, -mp_y],
                dtype=np.float64,
            )
            points.append(ros_point)

        return points

    def _compute_joint_position(
        self,
        config: JointConfig,
        landmarks: list[np.ndarray],
    ) -> float:
        a_idx, vertex_idx, c_idx = config.mediapipe_landmarks
        a = landmarks[a_idx]
        vertex = landmarks[vertex_idx]
        c = landmarks[c_idx]

        v1 = a - vertex
        v2 = c - vertex

        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product <= 1e-9 or np.isnan(norm_product):
            angle = 0.0
        else:
            cosine = float(np.dot(v1, v2) / norm_product)
            cosine = float(np.clip(cosine, -1.0, 1.0))
            angle = float(math.acos(cosine))

        if config.invert:
            angle = -angle

        angle += config.offset
        return float(np.clip(angle, config.minimum, config.maximum))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RetargetingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
