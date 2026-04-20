"""Low-latency full-body retargeting node using MoveIt2 IK service calls.

Subscribes:
- /mediapipe/pose_world_landmarks (std_msgs/Float32MultiArray)

Publishes:
- /op3/joint_commands (sensor_msgs/JointState)

This node is designed to preserve compatibility with ros2_control/action_dispatcher.py
by always publishing the 20 OP3 joints in controller order.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

try:
    from moveit_msgs.msg import MoveItErrorCodes
    from moveit_msgs.srv import GetPositionIK

    MOVEIT_AVAILABLE = True
except Exception:
    MOVEIT_AVAILABLE = False


EXPECTED_FLOAT_COUNT = 33 * 4
LANDMARK_COUNT = 33
VALUES_PER_LANDMARK = 4

# Must match ros2_control/forward_command_config.yaml order exactly.
OP3_JOINT_ORDER = [
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


@dataclass(frozen=True)
class GroupTarget:
    group_name: str
    landmark_index: int
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0


class MoveItIKRetargetingNode(Node):
    """Convert MediaPipe landmarks to full-body OP3 joint targets using MoveIt IK."""

    def __init__(self) -> None:
        super().__init__("moveit_ik_retargeting_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("ik_timeout_sec", 0.006)
        self.declare_parameter("use_ik", True)
        self.declare_parameter("position_scale", 0.45)
        self.declare_parameter("publish_rate_hz", 30.0)

        self._group_targets = [
            GroupTarget(group_name="left_arm", landmark_index=15, y_offset=0.02),
            GroupTarget(group_name="right_arm", landmark_index=16, y_offset=-0.02),
            GroupTarget(group_name="left_leg", landmark_index=27),
            GroupTarget(group_name="right_leg", landmark_index=28),
        ]

        self._base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self._ik_timeout_sec = (
            self.get_parameter("ik_timeout_sec").get_parameter_value().double_value
        )
        self._use_ik = self.get_parameter("use_ik").get_parameter_value().bool_value
        self._position_scale = (
            self.get_parameter("position_scale").get_parameter_value().double_value
        )
        self._publish_period = 1.0 / max(
            1.0,
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value,
        )

        self._publisher = self.create_publisher(JointState, "/op3/joint_commands", 10)
        self._subscription = self.create_subscription(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            self._landmarks_callback,
            10,
        )

        self._joint_positions = {joint_name: 0.0 for joint_name in OP3_JOINT_ORDER}
        self._last_publish_time = 0.0
        self._last_timing_log = time.monotonic()

        self._ik_client = None
        if MOVEIT_AVAILABLE and self._use_ik:
            ik_service = self.get_parameter("ik_service").get_parameter_value().string_value
            self._ik_client = self.create_client(GetPositionIK, ik_service)
            self.get_logger().info(
                f"Waiting briefly for MoveIt IK service: {ik_service}"
            )
            self._ik_client.wait_for_service(timeout_sec=1.0)
        elif self._use_ik and not MOVEIT_AVAILABLE:
            self.get_logger().warning(
                "moveit_msgs is not available; falling back to heuristic retargeting"
            )

        self.get_logger().info(
            "moveit_ik_retargeting_node started "
            f"(IK enabled={self._use_ik and MOVEIT_AVAILABLE})"
        )

    def _landmarks_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != EXPECTED_FLOAT_COUNT:
            self.get_logger().warning(
                f"Expected {EXPECTED_FLOAT_COUNT} floats, got {len(msg.data)}",
                throttle_duration_sec=2.0,
            )
            return

        now = time.perf_counter()
        if (now - self._last_publish_time) < self._publish_period:
            return

        start = time.perf_counter()
        landmarks = self._unpack_landmarks(msg.data)

        if self._use_ik and MOVEIT_AVAILABLE and self._ik_client is not None:
            self._run_moveit_ik(landmarks)
        else:
            self._run_heuristic_fallback(landmarks)

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(OP3_JOINT_ORDER)
        joint_state.position = [self._joint_positions[name] for name in OP3_JOINT_ORDER]
        self._publisher.publish(joint_state)
        self._last_publish_time = now

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        monotonic_now = time.monotonic()
        if monotonic_now - self._last_timing_log >= 3.0:
            self.get_logger().info(f"Retarget+IK loop time: {elapsed_ms:.2f} ms")
            self._last_timing_log = monotonic_now

    def _unpack_landmarks(self, flat_data: list[float]) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        for index in range(LANDMARK_COUNT):
            base = index * VALUES_PER_LANDMARK
            mp_x = float(flat_data[base])
            mp_y = float(flat_data[base + 1])
            mp_z = float(flat_data[base + 2])

            points.append(np.array([-mp_z, -mp_x, -mp_y], dtype=np.float64))
        return points

    def _run_moveit_ik(self, landmarks: list[np.ndarray]) -> None:
        solved_group_count = 0
        for group_target in self._group_targets:
            goal = landmarks[group_target.landmark_index] * self._position_scale
            goal = np.array(
                [
                    float(goal[0] + group_target.x_offset),
                    float(goal[1] + group_target.y_offset),
                    float(goal[2] + group_target.z_offset),
                ],
                dtype=np.float64,
            )

            solution = self._solve_group_ik(group_target.group_name, goal)
            if solution:
                self._joint_positions.update(solution)
                solved_group_count += 1

        if solved_group_count == 0:
            self._run_heuristic_fallback(landmarks)
            return

        self._update_head_from_face_landmarks(landmarks)

    def _solve_group_ik(self, group_name: str, target_xyz: np.ndarray) -> dict[str, float] | None:
        if self._ik_client is None:
            return None
        if not self._ik_client.service_is_ready():
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = group_name
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(
            sec=int(self._ik_timeout_sec),
            nanosec=int((self._ik_timeout_sec % 1.0) * 1_000_000_000),
        )

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._base_frame
        pose.pose.position.x = float(target_xyz[0])
        pose.pose.position.y = float(target_xyz[1])
        pose.pose.position.z = float(target_xyz[2])
        pose.pose.orientation.w = 1.0

        request.ik_request.pose_stamped = pose

        request.ik_request.robot_state.joint_state.name = list(OP3_JOINT_ORDER)
        request.ik_request.robot_state.joint_state.position = [
            self._joint_positions[name] for name in OP3_JOINT_ORDER
        ]

        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._ik_timeout_sec + 0.004)
        if not future.done() or future.result() is None:
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        solved = {}
        for name, position in zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ):
            if name in self._joint_positions:
                solved[name] = float(position)

        return solved

    def _run_heuristic_fallback(self, landmarks: list[np.ndarray]) -> None:
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_elbow = landmarks[13]
        r_elbow = landmarks[14]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_knee = landmarks[25]
        r_knee = landmarks[26]
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        l_foot_index = landmarks[31]
        r_foot_index = landmarks[32]

        shoulder_line = l_shoulder - r_shoulder
        hip_line = l_hip - r_hip
        body_up = (l_shoulder + r_shoulder) * 0.5 - (l_hip + r_hip) * 0.5

        self._joint_positions["l_sho_pitch"] = float(np.clip((l_wrist[2] - l_shoulder[2]) * 2.0, -2.2, 2.2))
        self._joint_positions["r_sho_pitch"] = float(np.clip((r_wrist[2] - r_shoulder[2]) * 2.0, -2.2, 2.2))
        self._joint_positions["l_sho_roll"] = float(np.clip((l_wrist[1] - l_shoulder[1]) * 2.0, -2.0, 2.0))
        self._joint_positions["r_sho_roll"] = float(np.clip((r_shoulder[1] - r_wrist[1]) * 2.0, -2.0, 2.0))

        self._joint_positions["l_el"] = self._angle_from_points(l_shoulder, l_elbow, l_wrist)
        self._joint_positions["r_el"] = self._angle_from_points(r_shoulder, r_elbow, r_wrist)

        self._joint_positions["l_hip_pitch"] = float(np.clip((l_knee[2] - l_hip[2]) * 1.8, -1.8, 1.8))
        self._joint_positions["r_hip_pitch"] = float(np.clip((r_knee[2] - r_hip[2]) * 1.8, -1.8, 1.8))
        self._joint_positions["l_hip_roll"] = float(np.clip((l_knee[1] - l_hip[1]) * 1.4, -1.2, 1.2))
        self._joint_positions["r_hip_roll"] = float(np.clip((r_hip[1] - r_knee[1]) * 1.4, -1.2, 1.2))

        body_twist = self._signed_angle_around_axis(hip_line, shoulder_line, body_up)
        hip_yaw = float(np.clip(body_twist * 0.9, -1.1, 1.1))
        self._joint_positions["l_hip_yaw"] = hip_yaw
        self._joint_positions["r_hip_yaw"] = -hip_yaw

        self._joint_positions["l_knee"] = self._angle_from_points(l_hip, l_knee, l_ankle)
        self._joint_positions["r_knee"] = self._angle_from_points(r_hip, r_knee, r_ankle)

        self._joint_positions["l_ank_pitch"] = float(np.clip((l_foot_index[2] - l_ankle[2]) * 2.0, -1.4, 1.4))
        self._joint_positions["r_ank_pitch"] = float(np.clip((r_foot_index[2] - r_ankle[2]) * 2.0, -1.4, 1.4))

        l_leg_roll_term = (l_ankle[1] - l_knee[1]) * 0.8
        r_leg_roll_term = (r_knee[1] - r_ankle[1]) * 0.8
        l_foot_roll_term = (l_foot_index[1] - l_ankle[1]) * 1.0
        r_foot_roll_term = (r_ankle[1] - r_foot_index[1]) * 1.0
        self._joint_positions["l_ank_roll"] = float(np.clip(l_leg_roll_term + l_foot_roll_term, -1.0, 1.0))
        self._joint_positions["r_ank_roll"] = float(np.clip(r_leg_roll_term + r_foot_roll_term, -1.0, 1.0))

        self._update_head_from_face_landmarks(landmarks)

    def _update_head_from_face_landmarks(self, landmarks: list[np.ndarray]) -> None:
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        nose = landmarks[0]

        shoulder_center = (landmarks[11] + landmarks[12]) * 0.5
        ear_delta = left_ear - right_ear

        head_pan = float(np.clip(math.atan2(ear_delta[0], max(1e-5, abs(ear_delta[1]))), -1.4, 1.4))
        head_tilt = float(np.clip((nose[2] - shoulder_center[2]) * 1.8, -1.0, 1.0))

        self._joint_positions["head_pan"] = head_pan
        self._joint_positions["head_tilt"] = head_tilt

    def _angle_from_points(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b

        norm_product = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm_product <= 1e-8 or math.isnan(norm_product):
            return 0.0

        cosine = float(np.dot(v1, v2) / norm_product)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.clip(math.acos(cosine), -2.4, 2.4))

    def _signed_angle_around_axis(
        self,
        from_vec: np.ndarray,
        to_vec: np.ndarray,
        axis: np.ndarray,
    ) -> float:
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            return 0.0

        unit_axis = axis / axis_norm
        from_proj = from_vec - unit_axis * np.dot(from_vec, unit_axis)
        to_proj = to_vec - unit_axis * np.dot(to_vec, unit_axis)

        from_norm = float(np.linalg.norm(from_proj))
        to_norm = float(np.linalg.norm(to_proj))
        if from_norm <= 1e-8 or to_norm <= 1e-8:
            return 0.0

        from_unit = from_proj / from_norm
        to_unit = to_proj / to_norm

        sine = float(np.dot(np.cross(from_unit, to_unit), unit_axis))
        cosine = float(np.dot(from_unit, to_unit))
        return float(math.atan2(sine, cosine))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveItIKRetargetingNode()

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
