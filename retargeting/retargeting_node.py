"""Retargeting: MediaPipe world landmarks → OP3 joint angles via body-relative decomposition."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
from one_euro_filter import OneEuroFilter


EXPECTED = 33 * 4

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
NOSE = 0
L_EAR,      R_EAR      = 7,  8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
L_FOOT,     R_FOOT     = 31, 32

# ── Joint limits [rad] ─────────────────────────────────────────────────────────
LIMITS: dict[str, tuple[float, float]] = {
    "head_pan":    (-1.5,  1.5),
    "head_tilt":   (-1.0,  1.0),
    "l_sho_pitch": (-2.0,  2.0),
    "r_sho_pitch": (-2.0,  2.0),
    "l_sho_roll":  (-1.6,  1.6),
    "r_sho_roll":  (-1.6,  1.6),
    "l_el":        ( 0.0,  2.4),
    "r_el":        ( 0.0,  2.4),
    "l_hip_pitch": (-1.8,  1.8),
    "r_hip_pitch": (-1.8,  1.8),
    "l_hip_roll":  (-0.8,  0.8),
    "r_hip_roll":  (-0.8,  0.8),
    "l_knee":      ( 0.0,  2.1),
    "r_knee":      ( 0.0,  2.1),
    "l_ank_pitch": (-1.0,  1.0),
    "r_ank_pitch": (-1.0,  1.0),
}
JOINT_NAMES = list(LIMITS.keys())

# Default values when landmarks are not visible — keeps the robot in a
# natural standing pose rather than collapsing to all-zero joints.
STANDING_DEFAULTS: dict[str, float] = {
    "head_pan":     0.0,
    "head_tilt":    0.0,
    "l_sho_pitch":  0.0,
    "r_sho_pitch":  0.0,
    "l_sho_roll":  -0.3,
    "r_sho_roll":   0.3,
    "l_el":         0.0,
    "r_el":         0.0,
    "l_hip_pitch": -0.57,
    "r_hip_pitch":  0.57,
    "l_hip_roll":   0.0,
    "r_hip_roll":   0.0,
    "l_knee":       1.2,
    "r_knee":       1.2,
    "l_ank_pitch":  0.62,
    "r_ank_pitch": -0.62,
}
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
NEUTRAL_COMMAND = {name: 0.0 for name in CONTROLLER_JOINTS}


# ── Math helpers ───────────────────────────────────────────────────────────────

def _n(v: np.ndarray) -> np.ndarray:
    """Normalize; returns zero vector if near-zero."""
    mag = np.linalg.norm(v)
    return v / mag if mag > 1e-9 else np.zeros(3)


def _bend(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex between rays vertex→a and vertex→c, in [0, π]."""
    v1, v2 = a - vertex, c - vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    return float(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def _ok(lm: list[np.ndarray], *indices: int) -> bool:
    """True when none of the requested landmarks contain NaN."""
    return all(not np.isnan(lm[i]).any() for i in indices)


# ── Core retargeting ───────────────────────────────────────────────────────────

def compute_joints(lm: list[np.ndarray]) -> dict[str, float]:
    """
    Build a body-fixed frame from torso landmarks, then decompose each limb
    segment into that frame to produce true pitch and roll angles.

    Body frame (person-relative):
      up    = hip-midpoint → shoulder-midpoint
      right = left-shoulder → right-shoulder  (person's right)
      fwd   = cross(right, up)                (toward camera)

    Shoulder pitch  = arm in the sagittal plane  (fwd / up)
    Shoulder roll   = arm in the frontal  plane  (right / up)
    Hip pitch/roll  = thigh decomposed the same way
    """
    joints: dict[str, float] = dict(STANDING_DEFAULTS)

    def clamp(name: str, val: float) -> float:
        lo, hi = LIMITS[name]
        return float(np.clip(val, lo, hi))

    # Body frame requires core torso landmarks
    if not _ok(lm, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return joints

    shoulder_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2.0
    hip_mid      = (lm[L_HIP]      + lm[R_HIP])      / 2.0

    up    = _n(shoulder_mid - hip_mid)
    right = _n(lm[R_SHOULDER] - lm[L_SHOULDER])
    fwd   = _n(np.cross(right, up))

    # ── Head ──────────────────────────────────────────────────────────────────
    if _ok(lm, NOSE):
        h = lm[NOSE] - shoulder_mid
        joints["head_pan"]  = clamp("head_pan",  math.atan2( np.dot(h, right), np.dot(h, up)))
        joints["head_tilt"] = clamp("head_tilt", math.atan2(-np.dot(h, fwd),   np.dot(h, up)) * 0.5)

    # ── Left arm ──────────────────────────────────────────────────────────────
    if _ok(lm, L_SHOULDER, L_ELBOW):
        la = lm[L_ELBOW] - lm[L_SHOULDER]
        joints["l_sho_pitch"] = clamp("l_sho_pitch",  math.atan2( np.dot(la, fwd),   -np.dot(la, up)))
        joints["l_sho_roll"]  = clamp("l_sho_roll",   math.atan2(-np.dot(la, right), -np.dot(la, up)))

    if _ok(lm, L_SHOULDER, L_ELBOW, L_WRIST):
        joints["l_el"] = clamp("l_el", math.pi - _bend(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]))

    # ── Right arm (same sign convention as right hip — no invert) ─────────────
    if _ok(lm, R_SHOULDER, R_ELBOW):
        ra = lm[R_ELBOW] - lm[R_SHOULDER]
        joints["r_sho_pitch"] = clamp("r_sho_pitch", math.atan2( np.dot(ra, fwd),   -np.dot(ra, up)))
        joints["r_sho_roll"]  = clamp("r_sho_roll",  math.atan2( np.dot(ra, right), -np.dot(ra, up)))

    if _ok(lm, R_SHOULDER, R_ELBOW, R_WRIST):
        joints["r_el"] = clamp("r_el", math.pi - _bend(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]))

    # ── Left leg ──────────────────────────────────────────────────────────────
    if _ok(lm, L_HIP, L_KNEE):
        lt = lm[L_KNEE] - lm[L_HIP]
        joints["l_hip_pitch"] = clamp("l_hip_pitch",  math.atan2( np.dot(lt, fwd),   -np.dot(lt, up)))
        joints["l_hip_roll"]  = clamp("l_hip_roll",   math.atan2(-np.dot(lt, right), -np.dot(lt, up)))

    if _ok(lm, L_HIP, L_KNEE, L_ANKLE):
        # π when straight, decreases when bent → subtract from π so 0=straight, +ve=bent
        joints["l_knee"] = clamp("l_knee", math.pi - _bend(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]))

    if _ok(lm, L_KNEE, L_ANKLE, L_FOOT):
        # foot perpendicular to leg = π/2 → neutral ankle; offset by π/2 so 0=neutral
        joints["l_ank_pitch"] = clamp("l_ank_pitch", _bend(lm[L_KNEE], lm[L_ANKLE], lm[L_FOOT]) - math.pi / 2.0)

    # ── Right leg ─────────────────────────────────────────────────────────────
    if _ok(lm, R_HIP, R_KNEE):
        rt = lm[R_KNEE] - lm[R_HIP]
        joints["r_hip_pitch"] = clamp("r_hip_pitch", math.atan2( np.dot(rt, fwd),   -np.dot(rt, up)))
        joints["r_hip_roll"]  = clamp("r_hip_roll",  math.atan2( np.dot(rt, right), -np.dot(rt, up)))

    if _ok(lm, R_HIP, R_KNEE, R_ANKLE):
        joints["r_knee"] = clamp("r_knee", math.pi - _bend(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]))

    if _ok(lm, R_KNEE, R_ANKLE, R_FOOT):
        joints["r_ank_pitch"] = clamp("r_ank_pitch", _bend(lm[R_KNEE], lm[R_ANKLE], lm[R_FOOT]) - math.pi / 2.0)

    return joints


def build_controller_command(joints: dict[str, float]) -> dict[str, float]:
    """Expand the retargeted subset into the full OP3 controller joint set."""
    command = dict(NEUTRAL_COMMAND)
    command.update(joints)
    return command


# ── ROS 2 node ─────────────────────────────────────────────────────────────────

class RetargetingNode(Node):
    def __init__(self) -> None:
        super().__init__("retargeting_node")

        self._publisher = self.create_publisher(JointState, "/op3/joint_commands", 10)
        self._subscription = self.create_subscription(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            self._callback,
            10,
        )

        # One-Euro filters on output angles — smooths jitter without adding lag
        self._filters: dict[str, OneEuroFilter] = {
            name: OneEuroFilter(min_cutoff=0.5, beta=0.05)
            for name in JOINT_NAMES
        }
        self._last_log = time.monotonic()
        self.get_logger().info("retargeting_node started — 16 joints, body-relative decomposition")

    def _callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != EXPECTED:
            return

        raw = np.asarray(msg.data, dtype=np.float64).reshape(33, 4)
        lm = [raw[i, :3] for i in range(33)]

        joints = compute_joints(lm)

        t = time.monotonic()
        smoothed = {name: self._filters[name](t, angle) for name, angle in joints.items()}

        command = build_controller_command(smoothed)

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(CONTROLLER_JOINTS)
        out.position = [float(command[name]) for name in CONTROLLER_JOINTS]
        self._publisher.publish(out)

        if t - self._last_log >= 5.0:
            self.get_logger().info("Retargeting running")
            self._last_log = t


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RetargetingNode()
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
