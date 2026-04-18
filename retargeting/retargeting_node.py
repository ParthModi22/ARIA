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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coordinate_transform import (
    norm as _n,
    canonical_up as _canonical_up,
    compute_body_frame,
    NOSE,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW,    R_ELBOW,
    L_WRIST,    R_WRIST,
    L_HIP,      R_HIP,
    L_KNEE,     R_KNEE,
    L_ANKLE,    R_ANKLE,
    L_FOOT,     R_FOOT,
)


EXPECTED = 33 * 4
MIME_TOPIC = "/mediapipe/pose_landmarks"

# Landmark indices imported from coordinate_transform (single source of truth).

# ── Visibility thresholds ──────────────────────────────────────────────────────
# Raised/moving arms can have vis ~0.2, so keep upper-body threshold low.
# Ankles/feet are unreliable when legs are partly out of frame — use strict threshold.
VIS_UPPER  = 0.20   # torso, shoulder, elbow, wrist
VIS_LOWER  = 0.20   # hip, knee
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

# Stage-1 mimic defaults keep the robot's arms relaxed unless the camera pose
# clearly asks for arm motion. The official OP3 shoulder-roll defaults look too
# much like "arms out" for the demo.
MIME_STANDING_DEFAULTS: dict[str, float] = {
    **STANDING_DEFAULTS,
    "l_sho_roll": 0.0,
    "r_sho_roll": 0.0,
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
ENABLE_LEG_TRACKING = True
DEBUG_UPPER_BODY_ONLY = False
ENABLE_2D_UPPER_BODY_MIME = True

MIME_ARM_DOWN_DY = 0.06
MIME_ARM_DOWN_RATIO = 0.55
MIME_SHOULDER_ROLL_GAIN = 0.95
MIME_SHOULDER_PITCH_GAIN = 1.6
MIME_SHOULDER_PITCH_DEADBAND = 0.08
MIME_SHOULDER_PITCH_LIMIT = 0.65
LEFT_SHOULDER_PITCH_SIGN = 1.0
RIGHT_SHOULDER_PITCH_SIGN = -1.0
LEFT_ELBOW_SIGN = 1.0
RIGHT_ELBOW_SIGN = 1.0


# ── Math helpers ───────────────────────────────────────────────────────────────

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


def _camera_roll(segment: np.ndarray, right: np.ndarray, up: np.ndarray) -> float:
    """Simple arm lift from shoulder using a camera-aligned frame."""
    norm = np.linalg.norm(segment)
    if norm <= 1e-9:
        return 0.0
    unit = segment / norm
    return float(math.atan2(np.dot(unit, right), -np.dot(unit, up)))


def _angle_at(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """2D/3D angle at vertex, in radians."""
    v1 = a - vertex
    v2 = c - vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 1e-9:
        return 0.0
    cos_angle = float(np.dot(v1, v2) / denom)
    return float(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


def _arm_lift_from_image(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """
    Frontal-camera arm lift angle from image landmarks.

    MediaPipe image coordinates have +x right and +y down.  This returns near 0
    for an arm hanging down and near pi/2 for T-pose or hands-up style poses.
    """
    v = elbow[:2] - shoulder[:2]
    if np.linalg.norm(v) <= 1e-9:
        return 0.0

    dx = abs(float(v[0]))
    dy = float(v[1])
    if dy > MIME_ARM_DOWN_DY and dx / max(dy, 1e-3) < MIME_ARM_DOWN_RATIO:
        return 0.0
    if dy < 0.0:
        # Elbow above shoulder: visually this should look like a high arm.
        return math.pi / 2.0
    return float(math.atan2(dx, max(dy, 1e-3)))


def _arm_is_down_in_image(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    shoulder_width: float,
) -> bool:
    """True for the common relaxed pose: elbow and wrist below the shoulder."""
    elbow_below = float(elbow[1] - shoulder[1]) > MIME_ARM_DOWN_DY
    wrist_below = float(wrist[1] - shoulder[1]) > MIME_ARM_DOWN_DY * 2.0
    wrist_near_body = abs(float(wrist[0] - shoulder[0])) < shoulder_width * 0.75
    return elbow_below and wrist_below and wrist_near_body


def _arm_pitch_from_image_depth(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Front/back shoulder pitch from MediaPipe image landmark depth.

    In MediaPipe pose landmarks, smaller z means closer to the camera.  A hand
    coming toward the camera therefore gives positive shoulder_z - wrist_z.
    """
    # When the hand is clearly above the shoulder, the visible gesture is an
    # up/side gesture; depth noise should not pull the arm forward/back.
    if float(wrist[1] - shoulder[1]) < -MIME_ARM_DOWN_DY:
        return 0.0

    depth = float(shoulder[2] - 0.5 * (elbow[2] + wrist[2]))
    if abs(depth) < MIME_SHOULDER_PITCH_DEADBAND:
        return 0.0
    return float(np.clip(
        MIME_SHOULDER_PITCH_GAIN * depth,
        -MIME_SHOULDER_PITCH_LIMIT,
        MIME_SHOULDER_PITCH_LIMIT,
    ))


def compute_upper_body_mime_joints(raw: np.ndarray) -> dict[str, float]:
    """
    Stage 1 hackathon mimicry: image-space head + arms only.

    This intentionally avoids lower-body/free-balance retargeting.  It follows
    the successful PyBullet reference pattern: compute simple 2D angles, then
    command a small stable subset of joints while the robot stands.
    """
    joints: dict[str, float] = dict(MIME_STANDING_DEFAULTS)
    lm = _make_lm(raw)

    def clamp(name: str, value: float) -> float:
        lo, hi = LIMITS[name]
        return float(np.clip(value, lo, hi))

    if not _ok(lm, L_SHOULDER, R_SHOULDER):
        return joints

    l_shoulder = lm[L_SHOULDER]
    r_shoulder = lm[R_SHOULDER]
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm((r_shoulder - l_shoulder)[:2]))
    if shoulder_width <= 1e-6:
        shoulder_width = 1.0

    if _ok(lm, NOSE):
        nose_delta = lm[NOSE][:2] - shoulder_mid[:2]
        joints["head_pan"] = clamp("head_pan", 0.9 * float(nose_delta[0]) / shoulder_width)
        # Keep tilt conservative; large head tilt targets can distract from arm debugging.
        joints["head_tilt"] = clamp("head_tilt", -0.25 * float(nose_delta[1]) / shoulder_width)

    if _ok(lm, L_SHOULDER, L_ELBOW, L_WRIST):
        if _arm_is_down_in_image(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST], shoulder_width):
            joints["l_sho_roll"] = 0.0
            joints["l_sho_pitch"] = 0.0
            joints["l_el"] = 0.0
        else:
            lift = max(
                _arm_lift_from_image(lm[L_SHOULDER], lm[L_ELBOW]),
                _arm_lift_from_image(lm[L_SHOULDER], lm[L_WRIST]),
            )
            joints["l_sho_roll"] = clamp("l_sho_roll", -MIME_SHOULDER_ROLL_GAIN * lift)
            pitch = _arm_pitch_from_image_depth(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST])
            joints["l_sho_pitch"] = clamp("l_sho_pitch", LEFT_SHOULDER_PITCH_SIGN * pitch)
            elbow_angle = _angle_at(lm[L_SHOULDER][:2], lm[L_ELBOW][:2], lm[L_WRIST][:2])
            joints["l_el"] = clamp("l_el", LEFT_ELBOW_SIGN * (math.pi - elbow_angle))

    if _ok(lm, R_SHOULDER, R_ELBOW, R_WRIST):
        if _arm_is_down_in_image(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST], shoulder_width):
            joints["r_sho_roll"] = 0.0
            joints["r_sho_pitch"] = 0.0
            joints["r_el"] = 0.0
        else:
            lift = max(
                _arm_lift_from_image(lm[R_SHOULDER], lm[R_ELBOW]),
                _arm_lift_from_image(lm[R_SHOULDER], lm[R_WRIST]),
            )
            joints["r_sho_roll"] = clamp("r_sho_roll", MIME_SHOULDER_ROLL_GAIN * lift)
            pitch = _arm_pitch_from_image_depth(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST])
            joints["r_sho_pitch"] = clamp("r_sho_pitch", RIGHT_SHOULDER_PITCH_SIGN * pitch)
            elbow_angle = _angle_at(lm[R_SHOULDER][:2], lm[R_ELBOW][:2], lm[R_WRIST][:2])
            joints["r_el"] = clamp("r_el", RIGHT_ELBOW_SIGN * (math.pi - elbow_angle))

    return joints


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
    if not _ok(lm, L_SHOULDER, R_SHOULDER):
        return joints

    sho_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2.0
    right = _n(lm[R_SHOULDER] - lm[L_SHOULDER])

    if DEBUG_UPPER_BODY_ONLY:
        up = _canonical_up()
        fwd = _n(np.cross(up, right))
        if np.linalg.norm(fwd) <= 1e-9:
            return joints

        if _ok(lm, NOSE):
            h = lm[NOSE] - sho_mid
            joints["head_pan"] = clamp("head_pan", math.atan2(np.dot(h, right), -np.dot(h, fwd)))
            joints["head_tilt"] = clamp("head_tilt", math.atan2(-np.dot(h, up), max(abs(np.dot(h, fwd)), 1e-9)) * 0.3)

        if _ok(lm, L_SHOULDER, L_ELBOW):
            la = lm[L_ELBOW] - lm[L_SHOULDER]
            joints["l_sho_roll"] = clamp("l_sho_roll", _camera_roll(la, right, up))

        if _ok(lm, R_SHOULDER, R_ELBOW):
            ra = lm[R_ELBOW] - lm[R_SHOULDER]
            joints["r_sho_roll"] = clamp("r_sho_roll", _camera_roll(ra, right, up))

        if _ok(lm, L_SHOULDER, L_ELBOW, L_WRIST):
            joints["l_el"] = clamp("l_el", math.pi - _bend(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]))

        if _ok(lm, R_SHOULDER, R_ELBOW, R_WRIST):
            joints["r_el"] = clamp("r_el", RIGHT_ELBOW_SIGN * (math.pi - _bend(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST])))

        return joints

    body_frame = compute_body_frame(lm)
    if body_frame is None:
        return joints
    right, up, fwd = body_frame   # right re-bound from compute_body_frame (same value)

    # ── Head ─────────────────────────────────────────────────────────────────
    if _ok(lm, NOSE):
        h = lm[NOSE] - sho_mid
        joints["head_pan"]  = clamp("head_pan",  math.atan2( np.dot(h, right), np.dot(h, up)))
        joints["head_tilt"] = clamp("head_tilt", math.atan2(-np.dot(h, fwd),   np.dot(h, up)) * 0.5)

    # ── Left arm ─────────────────────────────────────────────────────────────
    if _ok(lm, L_SHOULDER, L_ELBOW):
        la = lm[L_ELBOW] - lm[L_SHOULDER]
        # pitch: arm down→0, arm forward→+π/2, arm straight up→π (clamped 2.0)
        joints["l_sho_pitch"] = clamp("l_sho_pitch", math.atan2(np.dot(la, fwd), -np.dot(la, up)))
        # roll: arm to LEFT (la≈−right) → dot=−1 → negative = abduction (correct: init −0.3)
        joints["l_sho_roll"]  = clamp("l_sho_roll",  math.atan2( np.dot(la, right),  -np.dot(la, up)))

    if _ok(lm, L_SHOULDER, L_ELBOW, L_WRIST):
        # π when straight → 0, bends toward +2.4 as arm folds
        joints["l_el"] = clamp("l_el", math.pi - _bend(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]))

    # ── Right arm ────────────────────────────────────────────────────────────
    if _ok(lm, R_SHOULDER, R_ELBOW):
        ra = lm[R_ELBOW] - lm[R_SHOULDER]
        joints["r_sho_pitch"] = clamp("r_sho_pitch", math.atan2(np.dot(ra, fwd), -np.dot(ra, up)))
        # roll: arm to RIGHT (ra≈+right) → dot=+1 → positive = abduction (correct: init +0.3)
        joints["r_sho_roll"]  = clamp("r_sho_roll",  math.atan2( np.dot(ra, right),  -np.dot(ra, up)))

    if _ok(lm, R_SHOULDER, R_ELBOW, R_WRIST):
        joints["r_el"] = clamp("r_el", RIGHT_ELBOW_SIGN * (math.pi - _bend(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST])))

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
            MIME_TOPIC if ENABLE_2D_UPPER_BODY_MIME else "/mediapipe/pose_world_landmarks",
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
        if ENABLE_2D_UPPER_BODY_MIME:
            self.get_logger().info(
                f"retargeting_node ready — STAGE 1 2D upper-body mime from {MIME_TOPIC}"
            )
        else:
            self.get_logger().info(
                "retargeting_node ready — 3D world retargeting mode"
            )

    def _cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != EXPECTED:
            return

        raw = np.asarray(msg.data, dtype=np.float64).reshape(33, 4)
        joints = (
            compute_upper_body_mime_joints(raw)
            if ENABLE_2D_UPPER_BODY_MIME
            else compute_joints(raw)
        )
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
