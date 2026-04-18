"""Runtime verifier for no-Gazebo retargeting pipeline.

Checks:
- /op3/joint_commands publishes full 20-joint JointState set.
- /forward_position_controller/commands publishes 20-length Float64MultiArray.
- Optional rate checks for both topics.
"""

from __future__ import annotations

import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

EXPECTED_JOINTS = [
    "l_sho_pitch",
    "r_sho_pitch",
    "l_sho_roll",
    "r_sho_roll",
    "l_el",
    "r_el",
    "l_hip_yaw",
    "r_hip_yaw",
    "l_hip_roll",
    "r_hip_roll",
    "l_hip_pitch",
    "r_hip_pitch",
    "l_knee",
    "r_knee",
    "l_ank_pitch",
    "r_ank_pitch",
    "l_ank_roll",
    "r_ank_roll",
    "head_pan",
    "head_tilt",
]


class PipelineVerifier(Node):
    def __init__(self) -> None:
        super().__init__("pipeline_verifier")

        self.declare_parameter("timeout_sec", 12.0)
        self.declare_parameter("min_rate_hz", 10.0)
        self.declare_parameter("required_messages", 20)

        self._timeout_sec = self.get_parameter("timeout_sec").get_parameter_value().double_value
        self._min_rate_hz = self.get_parameter("min_rate_hz").get_parameter_value().double_value
        self._required_messages = (
            self.get_parameter("required_messages").get_parameter_value().integer_value
        )

        self._joint_msg_count = 0
        self._forward_msg_count = 0
        self._joint_timestamps: list[float] = []
        self._forward_timestamps: list[float] = []
        self._joint_set_ok = False
        self._forward_len_ok = False

        self.create_subscription(JointState, "/op3/joint_commands", self._on_joint_state, 10)
        self.create_subscription(
            Float64MultiArray,
            "/forward_position_controller/commands",
            self._on_forward_command,
            10,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        now = time.monotonic()
        self._joint_msg_count += 1
        self._joint_timestamps.append(now)

        names = set(msg.name)
        self._joint_set_ok = set(EXPECTED_JOINTS).issubset(names)

    def _on_forward_command(self, msg: Float64MultiArray) -> None:
        now = time.monotonic()
        self._forward_msg_count += 1
        self._forward_timestamps.append(now)
        self._forward_len_ok = len(msg.data) == len(EXPECTED_JOINTS)

    def run(self) -> int:
        start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start) < self._timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.2)
            if (
                self._joint_msg_count >= self._required_messages
                and self._forward_msg_count >= self._required_messages
            ):
                break

        joint_rate = self._estimate_rate(self._joint_timestamps)
        forward_rate = self._estimate_rate(self._forward_timestamps)

        checks = {
            "joint_messages": self._joint_msg_count >= self._required_messages,
            "forward_messages": self._forward_msg_count >= self._required_messages,
            "joint_set_ok": self._joint_set_ok,
            "forward_len_ok": self._forward_len_ok,
            "joint_rate_ok": joint_rate >= self._min_rate_hz,
            "forward_rate_ok": forward_rate >= self._min_rate_hz,
        }

        for key, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            self.get_logger().info(f"{status} {key}")

        self.get_logger().info(f"joint_rate_hz={joint_rate:.2f}")
        self.get_logger().info(f"forward_rate_hz={forward_rate:.2f}")

        return 0 if all(checks.values()) else 1

    def _estimate_rate(self, timestamps: list[float]) -> float:
        if len(timestamps) < 3:
            return 0.0

        intervals = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if (b - a) > 0.0]
        if not intervals:
            return 0.0

        mean_dt = statistics.mean(intervals)
        if mean_dt <= 0.0:
            return 0.0

        return 1.0 / mean_dt


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PipelineVerifier()
    code = 1
    try:
        code = node.run()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    raise SystemExit(code)


if __name__ == "__main__":
    main()
