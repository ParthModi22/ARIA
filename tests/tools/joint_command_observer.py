"""Observe /op3/joint_commands and print motion summary statistics.

Useful for no-Gazebo validation of complete-body output.
"""

from __future__ import annotations

import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointCommandObserver(Node):
    def __init__(self) -> None:
        super().__init__("joint_command_observer")

        self.declare_parameter("topic", "/op3/joint_commands")
        self.declare_parameter("duration_sec", 10.0)
        self.declare_parameter("min_messages", 30)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._duration_sec = self.get_parameter("duration_sec").get_parameter_value().double_value
        self._min_messages = self.get_parameter("min_messages").get_parameter_value().integer_value

        self._message_count = 0
        self._samples: dict[str, list[float]] = {}
        self._timestamps: list[float] = []

        self.create_subscription(JointState, topic, self._on_joint_state, 20)
        self.get_logger().info(
            f"Observing {topic} for {self._duration_sec:.1f}s (min msgs: {self._min_messages})"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self._message_count += 1
        self._timestamps.append(time.monotonic())

        for joint_name, position in zip(msg.name, msg.position):
            if joint_name not in self._samples:
                self._samples[joint_name] = []
            self._samples[joint_name].append(float(position))

    def run(self) -> int:
        start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start) < self._duration_sec:
            rclpy.spin_once(self, timeout_sec=0.2)

        if self._message_count < self._min_messages:
            self.get_logger().error(
                f"Not enough messages received: {self._message_count} < {self._min_messages}"
            )
            return 1

        estimated_rate = self._estimate_rate(self._timestamps)
        self.get_logger().info(f"Messages: {self._message_count}, Estimated rate: {estimated_rate:.2f} Hz")

        moving_joint_count = 0
        for joint_name in sorted(self._samples.keys()):
            data = self._samples[joint_name]
            if not data:
                continue
            minimum = min(data)
            maximum = max(data)
            amplitude = maximum - minimum
            rms = math.sqrt(statistics.mean(value * value for value in data))
            if amplitude > 0.03:
                moving_joint_count += 1
            self.get_logger().info(
                f"{joint_name:>12} min={minimum:+.3f} max={maximum:+.3f} amp={amplitude:.3f} rms={rms:.3f}"
            )

        self.get_logger().info(f"Moving joints (>0.03 rad amplitude): {moving_joint_count}")
        return 0

    def _estimate_rate(self, timestamps: list[float]) -> float:
        if len(timestamps) < 3:
            return 0.0
        intervals = [
            b - a
            for a, b in zip(timestamps[:-1], timestamps[1:])
            if (b - a) > 0.0
        ]
        if not intervals:
            return 0.0
        mean_dt = statistics.mean(intervals)
        if mean_dt <= 0.0:
            return 0.0
        return 1.0 / mean_dt


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = JointCommandObserver()
    exit_code = 1
    try:
        exit_code = node.run()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
