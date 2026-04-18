"""Bridge /op3/joint_commands (JointState) → /forward_position_controller/commands (Float64MultiArray)."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Must match the joint order in forward_command_config.yaml exactly.
CONTROLLER_JOINTS = [
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


class ActionDispatcher(Node):
    def __init__(self) -> None:
        super().__init__("action_dispatcher")

        self._positions: dict[str, float] = {name: 0.0 for name in CONTROLLER_JOINTS}

        self._publisher = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            10,
        )
        self._subscription = self.create_subscription(
            JointState,
            "/op3/joint_commands",
            self._joint_commands_callback,
            10,
        )

        self.get_logger().info(
            f"action_dispatcher started — controlling {len(CONTROLLER_JOINTS)} joints"
        )

    def _joint_commands_callback(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in self._positions:
                self._positions[name] = float(position)
            else:
                self.get_logger().warning(f"Unknown joint in command: {name}", throttle_duration_sec=5.0)

        out = Float64MultiArray()
        out.data = [self._positions[name] for name in CONTROLLER_JOINTS]
        self._publisher.publish(out)


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
