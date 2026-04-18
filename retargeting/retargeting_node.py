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
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
L_FOOT,     R_FOOT     = 31, 32

# ── Visibility thresholds ──────────────────────────────────────────────────────
# Raised/moving arms can have vis ~0.2, so keep upper-body threshold low.
# Ankles/feet are unreliable when legs are partly out of frame — use strict threshold.
VIS_UPPER  = 0.20   # torso, shoulder, elbow, wrist
VIS_LOWER  = 0.40   # hip, knee
VIS_FOOT   = 0.60   # ankle, foot (often occluded; bad data causes ankle flailing)

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

# ── OP3 official standing values ───────────────────────────────────────────────
# Used when landmarks are invisible/unreliable.  Matches xacro initial_value.
_Q6 = math.pi / 6   # 0.5236 rad (30°)
_Q3 = math.pi / 3   # 1.0472 rad (60°)

STANDING_DEFAULTS: dict[str, float] = {
    "head_pan":    0.0,
    "head_tilt":   0.0,
    "l_sho_pitch": 0.0,
    "r_sho_pitch": 0.0,
    "l_sho_roll":  -0.3,
    "r_sho_roll":   0.3,
    "l_el":         0.0,
    "r_el":         0.0,
    "l_hip_pitch": -_Q6,
    "r_hip_pitch":  _Q6,
    "l_hip_roll":   0.0,
    "r_hip_roll":   0.0,
    "l_knee":       _Q3,
    "r_knee":       _Q3,
    "l_ank_pitch":  _Q6,
    "r_ank_pitch": -_Q6,
}

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
_NEUTRAL = {name: 0.0 for name in CONTROLLER_JOINTS}
ENABLE_LEG_TRACKING = False


# ── Math helpers ───────────────────────────────────────────────────────────────

def _n(v: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(v)
    return v / m if m > 1e-9 else np.zeros(3)


def _bend(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at `vertex` in radians, in [0, π]."""
    v1, v2 = a - vertex, c - vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    return float(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def _ok(lm: list[np.ndarray], *idx: int) -> bool:
    return all(not np.isnan(lm[i]).any() for i in idx)


def _stable_pitch(segment: np.ndarray, fwd: np.ndarray) -> float:
    """
    Forward/backward pitch that stays near zero for arms hanging down or
    stretched sideways, instead of spuriously jumping toward +/- pi/2.
    """
    norm = np.linalg.norm(segment)
    if norm <= 1e-9:
        return 0.0
    unit = segment / norm
    seg_fwd = float(np.dot(unit, fwd))
    seg_non_fwd = math.sqrt(max(0.0, 1.0 - seg_fwd * seg_fwd))
    return float(math.atan2(seg_fwd, max(seg_non_fwd, 1e-9)))


# ── Visibility-filtered landmark array ────────────────────────────────────────

def _make_lm(raw: np.ndarray) -> list[np.ndarray]:
    """
    Apply per-group visibility thresholds and return a list of 33 landmark
    xyz vectors.  Low-visibility entries are replaced with NaN so _ok()
    guards fire and joints fall back to STANDING_DEFAULTS.
    """
    lm: list[np.ndarray] = []
    for i in range(33):
        vis = float(raw[i, 3])
        if i in (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT):
            threshold = VIS_FOOT
        elif i in (L_HIP, R_HIP, L_KNEE, R_KNEE):
            threshold = VIS_LOWER
        else:
            threshold = VIS_UPPER
        lm.append(raw[i, :3] if vis >= threshold else np.full(3, float("nan")))
    return lm


# ── Core retargeting ───────────────────────────────────────────────────────────

def compute_joints(raw: np.ndarray) -> dict[str, float]:
    """
    Map MediaPipe world landmarks to OP3 joint angles.

    Body frame (person-relative, right-handed):
      up    = hip-midpoint → shoulder-midpoint       (+y in MediaPipe world)
      right = L_shoulder  → R_shoulder               (+x in MediaPipe world)
      fwd   = cross(up, right)                       (-z = toward camera)

    Sign conventions verified against xacro initial_value:
      l_sho_roll  = -0.3  → arm slightly LEFT/outward  (negative = abduction)
      r_sho_roll  = +0.3  → arm slightly RIGHT/outward (positive = abduction)
      l_hip_pitch = -π/6  → thigh slightly forward     (negative = forward)
      r_hip_pitch = +π/6  → thigh slightly forward     (positive = forward)
      l_ank_pitch = +π/6  → dorsiflexion compensating hip lean
      r_ank_pitch = -π/6  → same, mirrored axis
    """
    joints: dict[str, float] = dict(STANDING_DEFAULTS)
    lm = _make_lm(raw)

    def clamp(name: str, v: float) -> float:
        lo, hi = LIMITS[name]
        return float(np.clip(v, lo, hi))

    # ── Body frame ────────────────────────────────────────────────────────────
    if not _ok(lm, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return joints

    sho_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2.0
    hip_mid = (lm[L_HIP]      + lm[R_HIP])      / 2.0

    up    = _n(sho_mid - hip_mid)
    right = _n(lm[R_SHOULDER] - lm[L_SHOULDER])
    fwd   = _n(np.cross(up, right))   # −z in MediaPipe world = toward camera

    # ── Head ─────────────────────────────────────────────────────────────────
    if _ok(lm, NOSE):
        h = lm[NOSE] - sho_mid
        joints["head_pan"]  = clamp("head_pan",  math.atan2( np.dot(h, right), np.dot(h, up)))
        joints["head_tilt"] = clamp("head_tilt", math.atan2(-np.dot(h, fwd),   np.dot(h, up)) * 0.5)

    # ── Left arm ─────────────────────────────────────────────────────────────
    if _ok(lm, L_SHOULDER, L_ELBOW):
        la = lm[L_ELBOW] - lm[L_SHOULDER]
        # Stable forward/backward pitch: T-pose stays near 0 instead of jumping upward.
        joints["l_sho_pitch"] = clamp("l_sho_pitch", _stable_pitch(la, fwd))
        # roll: arm to LEFT (la≈−right) → dot=−1 → negative = abduction (correct: init −0.3)
        joints["l_sho_roll"]  = clamp("l_sho_roll",  math.atan2( np.dot(la, right),  -np.dot(la, up)))

    if _ok(lm, L_SHOULDER, L_ELBOW, L_WRIST):
        # π when straight → 0, bends toward +2.4 as arm folds
        joints["l_el"] = clamp("l_el", math.pi - _bend(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]))

    # ── Right arm ────────────────────────────────────────────────────────────
    if _ok(lm, R_SHOULDER, R_ELBOW):
        ra = lm[R_ELBOW] - lm[R_SHOULDER]
        joints["r_sho_pitch"] = clamp("r_sho_pitch", _stable_pitch(ra, fwd))
        # roll: arm to RIGHT (ra≈+right) → dot=+1 → positive = abduction (correct: init +0.3)
        joints["r_sho_roll"]  = clamp("r_sho_roll",  math.atan2( np.dot(ra, right),  -np.dot(ra, up)))

    if _ok(lm, R_SHOULDER, R_ELBOW, R_WRIST):
        joints["r_el"] = clamp("r_el", math.pi - _bend(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]))

    # ── Legs — BOTH or neither ────────────────────────────────────────────────
    # Only compute legs when BOTH hips AND both knees are visible.
    # If just one leg is tracked the robot gets an asymmetric "raised knee" pose
    # because one side computes to ~0 while the other falls back to ±π/6.
    both_hips_knees = ENABLE_LEG_TRACKING and _ok(lm, L_HIP, L_KNEE, R_HIP, R_KNEE)

    if both_hips_knees:
        # Left thigh vector: knee − hip
        lt = lm[L_KNEE] - lm[L_HIP]
        # Negate fwd component so forward-leaning thigh → negative pitch (init = −π/6)
        joints["l_hip_pitch"] = clamp("l_hip_pitch", math.atan2(-np.dot(lt, fwd),   -np.dot(lt, up)))
        joints["l_hip_roll"]  = clamp("l_hip_roll",  math.atan2(-np.dot(lt, right),  -np.dot(lt, up)))

        # Right thigh vector
        rt = lm[R_KNEE] - lm[R_HIP]
        # Same forward lean → positive pitch (init = +π/6)
        joints["r_hip_pitch"] = clamp("r_hip_pitch", math.atan2( np.dot(rt, fwd),    -np.dot(rt, up)))
        joints["r_hip_roll"]  = clamp("r_hip_roll",  math.atan2( np.dot(rt, right),   -np.dot(rt, up)))

        # Knees — need ankle as well
        if _ok(lm, L_ANKLE):
            joints["l_knee"] = clamp("l_knee", math.pi - _bend(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]))
        if _ok(lm, R_ANKLE):
            joints["r_knee"] = clamp("r_knee", math.pi - _bend(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]))

        # Ankles — only when foot landmark is reliable (VIS_FOOT = 0.60)
        if _ok(lm, L_KNEE, L_ANKLE, L_FOOT):
            joints["l_ank_pitch"] = clamp("l_ank_pitch",
                _bend(lm[L_KNEE], lm[L_ANKLE], lm[L_FOOT]) - math.pi / 2.0)
        if _ok(lm, R_KNEE, R_ANKLE, R_FOOT):
            joints["r_ank_pitch"] = clamp("r_ank_pitch",
                math.pi / 2.0 - _bend(lm[R_KNEE], lm[R_ANKLE], lm[R_FOOT]))

    return joints


def build_command(joints: dict[str, float]) -> dict[str, float]:
    cmd = dict(_NEUTRAL)
    cmd.update(joints)
    return cmd


# ── ROS 2 node ─────────────────────────────────────────────────────────────────

class RetargetingNode(Node):
    def __init__(self) -> None:
        super().__init__("retargeting_node")

        self._pub = self.create_publisher(JointState, "/op3/joint_commands", 10)
        self.create_subscription(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            self._cb,
            10,
        )

        # One-Euro filter: min_cutoff=1.5 for fast response, beta=0.2 for speed adaptation
        self._filters: dict[str, OneEuroFilter] = {
            name: OneEuroFilter(min_cutoff=1.5, beta=0.2)
            for name in JOINT_NAMES
        }
        self._last_log = time.monotonic()
        self._leg_track_log = 0.0
        self.get_logger().info(
            "retargeting_node ready — upper VIS≥0.20  lower VIS≥0.40  foot VIS≥0.60"
        )

    def _cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != EXPECTED:
            return

        raw = np.asarray(msg.data, dtype=np.float64).reshape(33, 4)
        joints   = compute_joints(raw)
        t        = time.monotonic()
        smoothed = {name: self._filters[name](t, v) for name, v in joints.items()}
        cmd      = build_command(smoothed)

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name     = list(CONTROLLER_JOINTS)
        out.position = [float(cmd[n]) for n in CONTROLLER_JOINTS]
        self._pub.publish(out)

        # ── Diagnostic log every 2 s ──────────────────────────────────────────
        if t - self._last_log >= 2.0:
            lp  = smoothed["l_sho_pitch"]; rp  = smoothed["r_sho_pitch"]
            lr  = smoothed["l_sho_roll"];  rr  = smoothed["r_sho_roll"]
            le  = smoothed["l_el"];        re  = smoothed["r_el"]
            lhp = smoothed["l_hip_pitch"]; rhp = smoothed["r_hip_pitch"]
            lk  = smoothed["l_knee"];      rk  = smoothed["r_knee"]

            # Detect whether legs are being tracked (values differ from defaults)
            legs_active = (
                abs(lhp - STANDING_DEFAULTS["l_hip_pitch"]) > 0.05 or
                abs(rhp - STANDING_DEFAULTS["r_hip_pitch"]) > 0.05
            )

            self.get_logger().info(
                f"ARMS  l_pitch={lp:+.2f} r_pitch={rp:+.2f} "
                f"l_roll={lr:+.2f} r_roll={rr:+.2f} "
                f"l_el={le:.2f} r_el={re:.2f}  |  "
                f"LEGS {'TRACKING' if legs_active else 'DEFAULTS'} "
                f"l_hip={lhp:+.2f} r_hip={rhp:+.2f} "
                f"l_knee={lk:.2f} r_knee={rk:.2f}"
            )
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
