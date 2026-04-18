"""
MediaPipe → OP3 body-frame coordinate transforms.

MediaPipe World Landmark Coordinate System
==========================================
MediaPipe world_landmarks are person-centric and right-handed:

    +x  = person's RIGHT  (toward their right shoulder)
    +y  = UP              (toward the top of the head)
    +z  = BEHIND person   (positive z points away from the camera)

The person faces the −z direction (toward the camera).

REP-103 (ROS standard robot body frame)
========================================
    +x  = FORWARD
    +y  = LEFT
    +z  = UP

Transform: MediaPipe → REP-103
    ros_x = −mp_z    (forward  = toward camera  = MediaPipe −z)
    ros_y = −mp_x    (left     = person's left  = MediaPipe −x)
    ros_z =  mp_y    (up       = upward          = MediaPipe +y)

Body-Relative Frame (used by retargeting_node.py for joint angles)
===================================================================
Joint angles are computed in a PERSON-RELATIVE body frame so they are
independent of the camera position or how the person is oriented in the frame.

    right = unit(R_shoulder − L_shoulder)             # ≈ MediaPipe +x
    up    = unit(shoulder_midpoint − hip_midpoint)    # ≈ MediaPipe +y
    fwd   = cross(up, right)                          # ≈ MediaPipe −z (toward camera)

Why body-relative instead of world-frame?
    If the person turns or shifts sideways in front of the camera, world-frame
    projections change even though the person hasn't moved their arms relative
    to their body.  The body frame removes that positional bias.

Sign conventions (verified against OP3 xacro initial_value):
    l_sho_roll  = −0.3  → arm slightly outward to the LEFT  (negative = abduction)
    r_sho_roll  = +0.3  → arm slightly outward to the RIGHT (positive = abduction)
    l_hip_pitch = −π/6  → thigh slightly forward            (negative = forward)
    r_hip_pitch = +π/6  → thigh slightly forward            (positive = forward)
    l_ank_pitch = +π/6  → dorsiflexion compensating hip lean
    r_ank_pitch = −π/6  → same, mirrored axis convention
"""

from __future__ import annotations

import numpy as np


# ── MediaPipe pose landmark indices ───────────────────────────────────────────
# Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
NOSE         =  0
L_EAR        =  7
R_EAR        =  8
L_SHOULDER   = 11
R_SHOULDER   = 12
L_ELBOW      = 13
R_ELBOW      = 14
L_WRIST      = 15
R_WRIST      = 16
L_HIP        = 23
R_HIP        = 24
L_KNEE       = 25
R_KNEE       = 26
L_ANKLE      = 27
R_ANKLE      = 28
L_FOOT       = 31   # left foot index (heel/toe tip)
R_FOOT       = 32   # right foot index


# ── Vector math ───────────────────────────────────────────────────────────────

def norm(v: np.ndarray) -> np.ndarray:
    """Return unit vector, or zero vector if near-zero magnitude."""
    m = float(np.linalg.norm(v))
    return v / m if m > 1e-9 else np.zeros(3, dtype=np.float64)


def canonical_up() -> np.ndarray:
    """
    World-aligned fallback 'up' direction used when hip landmarks are invisible.

    MediaPipe world +y = up, so the canonical up is simply (0, 1, 0).
    """
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


# ── Coordinate transforms ─────────────────────────────────────────────────────

def mediapipe_to_rep103(
    mp_point: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Convert a MediaPipe world landmark (x, y, z) to ROS REP-103 (x, y, z).

    MediaPipe world:  +x = person's right, +y = up,      +z = behind person
    REP-103:          +x = forward,         +y = left,    +z = up

    Derivation:
        fwd (REP-103 +x) = toward camera  = MediaPipe −z  → ros_x = −mp_z
        left(REP-103 +y) = person's left  = MediaPipe −x  → ros_y = −mp_x
        up  (REP-103 +z) = upward         = MediaPipe +y  → ros_z =  mp_y

    Example — arm straight forward (MediaPipe z = −0.3, x ≈ 0, y ≈ 0):
        rep103 = (0.3, 0, 0) = forward ✓
    """
    mx, my, mz = mp_point
    return (-mz, -mx, my)


# ── Body frame ─────────────────────────────────────────────────────────────────

def compute_body_frame(
    lm: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute the person-relative body frame axes from a MediaPipe landmark list.

    Args:
        lm: List of 33 (x, y, z) ndarray landmarks.  Invisible landmarks are NaN.

    Returns:
        (right, up, fwd) unit vectors, or None if both shoulders are invisible
        (cannot form a body frame without at least the shoulder axis).

    Frame axes (all unit vectors):
        right = unit(R_shoulder − L_shoulder)          # ≈ MediaPipe +x
        up    = unit(shoulder_mid − hip_mid)            # ≈ MediaPipe +y
                 falls back to nose−shoulder or +y if hips are unseen
        fwd   = cross(up, right)                        # ≈ MediaPipe −z (toward camera)
    """
    l_sho = lm[L_SHOULDER]
    r_sho = lm[R_SHOULDER]
    if np.isnan(l_sho).any() or np.isnan(r_sho).any():
        return None  # cannot form frame without both shoulders

    right   = norm(r_sho - l_sho)
    sho_mid = (l_sho + r_sho) / 2.0

    # Prefer spine direction (shoulder_mid → hip_mid) for 'up'
    l_hip = lm[L_HIP]
    r_hip = lm[R_HIP]
    if not (np.isnan(l_hip).any() or np.isnan(r_hip).any()):
        hip_mid = (l_hip + r_hip) / 2.0
        up = norm(sho_mid - hip_mid)
    elif not np.isnan(lm[NOSE]).any():
        # Head above shoulders is a reasonable up proxy
        up = norm(lm[NOSE] - sho_mid)
    else:
        up = canonical_up()

    # Guard against degenerate up (person lying flat, or hips at same height as shoulders)
    if np.linalg.norm(up) <= 1e-9:
        up = canonical_up()

    fwd = norm(np.cross(up, right))  # −z in MediaPipe world = toward camera ✓
    if np.linalg.norm(fwd) <= 1e-9:
        # Fallback: recompute fwd using world-aligned up
        up  = canonical_up()
        fwd = norm(np.cross(up, right))
    if np.linalg.norm(fwd) <= 1e-9:
        # Degenerate: shoulders are perfectly vertical — cannot form a valid frame
        return None

    return right, up, fwd
