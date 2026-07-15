"""
Shared utilities for the Pose Exercise Recording & Playback system.
Contains common constants, angle calculation, and pose normalization functions.
"""

import math

# ─── Pose Topology ─────────────────────────────────────────────────────

POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24), (23, 24),  # torso
]

# Backward-compatible aliases for older modules.
HAND_CONNECTIONS = POSE_CONNECTIONS
FINGERTIP_IDS = []

ANGLE_JOINTS = [
    (23, 11, 13),  # left shoulder
    (11, 13, 15),  # left elbow
    (24, 12, 14),  # right shoulder
    (12, 14, 16),  # right elbow
]


# ─── Math Helpers ──────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Calculate the angle (degrees) at point b formed by segments ba and bc."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def compute_angles(points):
    """Compute all joint angles from 21 (x, y) landmark points."""
    return [
        calculate_angle(points[p1], points[p2], points[p3])
        for p1, p2, p3 in ANGLE_JOINTS
    ]


def compute_match_score(patient_angles, target_angles, threshold=20.0):
    """
    Compare patient angles to target angles.
    Returns (match_fraction, per_joint_matched list).
    threshold: max degrees difference for a joint to count as matched.
    """
    per_joint = [abs(p - t) < threshold for p, t in zip(patient_angles, target_angles)]
    fraction = sum(per_joint) / len(per_joint) if per_joint else 0.0
    return fraction, per_joint


def normalize_pose_landmarks(landmarks_xy):
    """
    Normalize 33 (x, y) landmarks relative to shoulder center and shoulder width.
    Returns (relative_points, pose_scale).
    """
    left_shoulder = landmarks_xy[11]
    right_shoulder = landmarks_xy[12]
    anchor_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    anchor_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    relative = [(x - anchor_x, y - anchor_y) for x, y in landmarks_xy]

    pose_scale = math.dist(left_shoulder, right_shoulder)
    if pose_scale < 1e-6:
        pose_scale = 1.0

    return relative, pose_scale


def normalize_hand_pose(landmarks_xy):
    return normalize_pose_landmarks(landmarks_xy)


def project_ghost(target_relative, target_scale, patient_anchor, patient_scale):
    """
    Project normalized target landmarks onto patient's pose frame.
    Anchors ghost to patient's shoulder center, scales to patient's shoulder width.
    Returns list of (x, y) pixel coordinates for the ghost pose.
    """
    scale = patient_scale / target_scale if target_scale > 1e-6 else 1.0
    wx, wy = patient_anchor
    return [
        (int(rx * scale + wx), int(ry * scale + wy))
        for rx, ry in target_relative
    ]
