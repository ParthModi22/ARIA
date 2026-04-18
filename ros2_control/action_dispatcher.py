"""Bridge /op3/joint_commands (JointState) → /forward_position_controller/commands (Float64MultiArray).

Publishes at 30 Hz regardless of retargeting rate so the controller always
receives commands and the robot doesn't lock into a stale pose.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Must match the joint order in forward_command_config.yaml exactly.
CONTROLLER_JOINTS = [
    "l_sho_pitch", "r_sho_pitch",
    "l_sho_roll",  "r_sho_roll",
    "l_el",        "r_el",
    "l_hip_yaw",   "r_hip_yaw",
    "l_hip_roll",  "r_hip_roll",
    "l_hip_pitch", "r_hip_pitch",
    "l_knee",      "r_knee",
    "l_ank_pitch", "r_ank_pitch",
    "l_ank_roll",  "r_ank_roll",
    "head_pan",    "head_tilt",
]

# Stable demo fallback pose. Retargeting overwrites these values when landmarks
# are available; this keeps the robot relaxed if perception briefly drops.
_Q6 = math.pi / 6   # 0.5236 rad
_Q3 = math.pi / 3   # 1.0472 rad
_ARM_DOWN_PITCH = -1.15

STANDING_POSE: dict[str, float] = {
    "l_sho_pitch":  _ARM_DOWN_PITCH,
    "r_sho_pitch":  _ARM_DOWN_PITCH,
    "l_sho_roll":   0.0,
    "r_sho_roll":   0.0,
    "l_el":         0.0,
    "r_el":         0.0,
    "l_hip_yaw":    0.0,
    "r_hip_yaw":    0.0,
    "l_hip_roll":   0.0,
    "r_hip_roll":   0.0,
    "l_hip_pitch": -_Q6,
    "r_hip_pitch":  _Q6,
    "l_knee":       _Q3,
    "r_knee":       _Q3,
    "l_ank_pitch":  _Q6,
    "r_ank_pitch": -_Q6,
    "l_ank_roll":   0.0,
    "r_ank_roll":   0.0,
    "head_pan":     0.0,
    "head_tilt":    0.0,
}


class ActionDispatcher(Node):
    def __init__(self) -> None:
        super().__init__("action_dispatcher")

        self._positions: dict[str, float] = dict(STANDING_POSE)

        self._pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            10,
        )
        self.create_subscription(
            JointState,
            "/op3/joint_commands",
            self._on_joint_commands,
            10,
        )

        # Publish at 30 Hz regardless of whether new retargeting data arrives.
        # This keeps the controller's command stream alive and prevents the
        # robot from locking in a stale pose when the perception pipeline
        # is slow or briefly drops frames.
        self._timer = self.create_timer(1.0 / 30.0, self._publish)

        self.get_logger().info(
            f"action_dispatcher ready — {len(CONTROLLER_JOINTS)} joints @ 30 Hz"
        )

    def _on_joint_commands(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self._positions:
                self._positions[name] = float(pos)
            else:
                self.get_logger().warning(
                    f"Unknown joint: {name}", throttle_duration_sec=10.0
                )

    def _publish(self) -> None:
        out = Float64MultiArray()
        out.data = [self._positions[name] for name in CONTROLLER_JOINTS]
        self._pub.publish(out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActionDispatcher()
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
