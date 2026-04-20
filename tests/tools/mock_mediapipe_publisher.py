"""Deterministic MediaPipe-like landmark publisher for no-Gazebo validation."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

LANDMARK_COUNT = 33
VALUES_PER_LANDMARK = 4


class MockMediaPipePublisher(Node):
    def __init__(self) -> None:
        super().__init__("mock_mediapipe_publisher")

        self.declare_parameter("topic", "/mediapipe/pose_world_landmarks")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("amplitude", 0.15)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self._amplitude = self.get_parameter("amplitude").get_parameter_value().double_value

        self._publisher = self.create_publisher(Float32MultiArray, topic, 10)
        self._tick = 0

        period = 1.0 / max(1.0, self._rate_hz)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(f"mock_mediapipe_publisher publishing to {topic} @ {self._rate_hz:.1f}Hz")

    def _on_timer(self) -> None:
        phase = self._tick * 0.08

        # Simple body template in normalized MediaPipe-like coordinates.
        points = [(0.0, 0.0, 0.0, 1.0) for _ in range(LANDMARK_COUNT)]

        # Torso anchors.
        points[11] = (-0.15, -0.10, 0.20, 1.0)  # left shoulder
        points[12] = (0.15, -0.10, 0.20, 1.0)   # right shoulder
        points[23] = (-0.10, 0.18, 0.05, 1.0)   # left hip
        points[24] = (0.10, 0.18, 0.05, 1.0)    # right hip

        arm_lift = self._amplitude * math.sin(phase)
        elbow_bend = 0.06 + self._amplitude * 0.3 * (1.0 + math.sin(phase + 0.7))

        # Arms.
        points[13] = (-0.26, -0.05 - arm_lift * 0.5, 0.18, 1.0)  # left elbow
        points[14] = (0.26, -0.05 + arm_lift * 0.5, 0.18, 1.0)   # right elbow
        points[15] = (-0.34, 0.02 - arm_lift, 0.15 - elbow_bend, 1.0)  # left wrist
        points[16] = (0.34, 0.02 + arm_lift, 0.15 - elbow_bend, 1.0)   # right wrist

        # Legs (small marching motion).
        step = self._amplitude * 0.5 * math.sin(phase + 1.2)
        points[25] = (-0.11, 0.36, -0.13 + step, 1.0)  # left knee
        points[26] = (0.11, 0.36, -0.13 - step, 1.0)   # right knee
        points[27] = (-0.11, 0.55, -0.36 + step, 1.0)  # left ankle
        points[28] = (0.11, 0.55, -0.36 - step, 1.0)   # right ankle

        # Face points for head pan/tilt estimation.
        points[0] = (0.0, -0.22 + 0.04 * math.sin(phase * 0.7), 0.32, 1.0)   # nose
        points[7] = (-0.06, -0.20, 0.30, 1.0)  # left ear
        points[8] = (0.06, -0.20, 0.30, 1.0)   # right ear

        flat = []
        for x, y, z, visibility in points:
            flat.extend([float(x), float(y), float(z), float(visibility)])

        msg = Float32MultiArray()
        msg.data = flat
        self._publisher.publish(msg)

        self._tick += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockMediaPipePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
