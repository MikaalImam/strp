"""
Guided Exercise Playback
=========================
The patient loads a recorded exercise and follows along.
A ghost hand shows the target pose. It only advances to the next
step when the patient matches the current pose closely enough.

Usage:
  python follow_exercise.py                     # lists available exercises
  python follow_exercise.py <exercise_name>     # loads directly

Controls:
  R     - Restart the exercise
  Q     - Quit
"""

import cv2
import mediapipe as mp
import time
import math
import json
import sys
import os
from mediapipe.tasks.python import vision

from exercise_utils import (
    HAND_CONNECTIONS, ANGLE_JOINTS, FINGERTIP_IDS,
    calculate_angle, compute_angles, compute_match_score,
    normalize_hand_pose, project_ghost
)

# ─── Config ────────────────────────────────────────────────────────────

MODEL_PATH = "../models/pose_landmarker_lite.task"
EXERCISES_DIR = "../recorded_exercises"

MATCH_THRESHOLD = 20    # degrees tolerance per joint
MATCH_REQUIRED = 0.75     # fraction of joints that must match (75%)
HOLD_DURATION = .5       # seconds the patient must hold a match before advancing

# Ghost hand colors (BGR)
GHOST_COLOR = (200, 150, 50)         # Soft blue
GHOST_LINE_COLOR = (180, 130, 40)    # Darker blue for connections
MATCHED_COLOR = (0, 230, 0)          # Green when matched
UNMATCHED_COLOR = (0, 80, 255)       # Orange-red when far

# ─── MediaPipe Setup ───────────────────────────────────────────────────

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=on_result
)

# ─── Exercise Loading ──────────────────────────────────────────────────

def list_exercises():
    """List all available exercise files."""
    if not os.path.isdir(EXERCISES_DIR):
        return []
    return [f.replace(".json", "") for f in os.listdir(EXERCISES_DIR) if f.endswith(".json")]


def load_exercise(name):
    """Load an exercise from JSON."""
    filepath = os.path.join(EXERCISES_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        print(f"[ERROR] Exercise file not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"[LOADED] '{data['name']}' — {data['num_keyframes']} keyframes")
    return data


def select_exercise():
    """Interactive exercise selection."""
    exercises = list_exercises()
    if not exercises:
        print("[ERROR] No exercises found. Record one first with record_exercise.py")
        sys.exit(1)

    print("\n  Available exercises:")
    for i, name in enumerate(exercises):
        print(f"    {i + 1}. {name}")
    print()

    choice = input("  Enter exercise number or name: ").strip()

    # Try as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(exercises):
            return exercises[idx]
    except ValueError:
        pass

    # Try as name
    if choice in exercises:
        return choice

    print(f"[ERROR] Invalid selection: {choice}")
    sys.exit(1)

# ─── Drawing Functions ─────────────────────────────────────────────────

def draw_ghost_hand(frame, ghost_points):
    """Draw the ghost (target) hand skeleton on frame."""
    # Draw connections
    overlay = frame.copy()
    for start, end in HAND_CONNECTIONS:
        if ghost_points[start] and ghost_points[end]:
            cv2.line(overlay, ghost_points[start], ghost_points[end],
                     GHOST_LINE_COLOR, 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Draw landmark dots
    for idx, pt in enumerate(ghost_points):
        if pt is None:
            continue
        # Outer glow
        overlay2 = frame.copy()
        cv2.circle(overlay2, pt, 12, GHOST_COLOR, -1)
        cv2.addWeighted(overlay2, 0.15, frame, 0.85, 0, frame)
        # Core dot
        cv2.circle(frame, pt, 6, GHOST_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_patient_hand(frame, points, per_joint_matched=None):
    """Draw the patient's actual hand with color-coded accuracy."""
    # Build per-landmark color based on joint matching
    landmark_colors = {}
    if per_joint_matched is not None:
        for i, (p1, p2, p3) in enumerate(ANGLE_JOINTS):
            color = MATCHED_COLOR if per_joint_matched[i] else UNMATCHED_COLOR
            # Color the middle joint (where angle is measured)
            landmark_colors[p2] = color

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (200, 200, 200), 2, cv2.LINE_AA)

    # Draw landmarks
    for idx, pt in enumerate(points):
        color = landmark_colors.get(idx, (0, 255, 0))
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_progress_bar(frame, current, total):
    """Draw a progress bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 8
    bar_y = 50

    # Background
    cv2.rectangle(frame, (20, bar_y), (w - 20, bar_y + bar_h), (40, 40, 40), -1)

    # Progress fill
    progress = current / total if total > 0 else 0
    fill_w = int((w - 40) * progress)
    cv2.rectangle(frame, (20, bar_y), (20 + fill_w, bar_y + bar_h), (0, 215, 255), -1)

    # Step markers
    if total > 1:
        for i in range(total):
            x = 20 + int((w - 40) * i / (total - 1))
            color = (0, 255, 0) if i < current else (80, 80, 80)
            cv2.circle(frame, (x, bar_y + bar_h // 2), 3, color, -1)


def draw_match_indicator(frame, match_frac):
    """Draw circular match percentage indicator."""
    h, w = frame.shape[:2]
    cx, cy = w - 60, 90
    radius = 35

    # Background circle
    cv2.circle(frame, (cx, cy), radius, (40, 40, 40), -1)

    # Arc showing match percentage
    angle = int(360 * match_frac)
    if match_frac >= MATCH_REQUIRED:
        color = (0, 230, 0)  # Green
    elif match_frac >= 0.5:
        color = (0, 200, 255)  # Yellow
    else:
        color = (0, 80, 255)  # Red

    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, color, 4, cv2.LINE_AA)

    # Percentage text
    pct = int(match_frac * 100)
    text = f"{pct}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Status message + color shown at the bottom of the HUD, keyed by state
STATE_MESSAGES = {
    "no_hand": ("Show your hand to the camera", (100, 100, 100)),
    "waiting": ("Match the ghost hand pose", (180, 180, 180)),
    "holding": ("Hold steady...", (0, 200, 255)),
    "matched": ("MATCHED! Moving to next step...", (0, 255, 0)),
    "complete": ("EXERCISE COMPLETE!", (0, 255, 0)),
}


def draw_hud(frame, exercise_name, current_step, total_steps, state, match_frac):
    """Draw the heads-up display."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(frame, f"EXERCISE: {exercise_name.upper()}", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Step {current_step + 1} / {total_steps}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Progress bar
    draw_progress_bar(frame, current_step, total_steps)

    # Match indicator
    draw_match_indicator(frame, match_frac)

    # State text
    msg, color = STATE_MESSAGES.get(state, STATE_MESSAGES["no_hand"])
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, msg, ((w - tw) // 2, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "[R] Restart   [Q] Quit", (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)


def draw_completion_screen(frame):
    """Draw a completion celebration overlay."""
    h, w = frame.shape[:2]

    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Completion text
    msg1 = "EXERCISE COMPLETE!"
    msg2 = "Great job! Press [R] to restart or [Q] to quit."

    (tw1, _), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    (tw2, _), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.putText(frame, msg1, ((w - tw1) // 2, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, msg2, ((w - tw2) // 2, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

# ─── Helpers ───────────────────────────────────────────────────────────

def get_ghost_at_default_position(keyframe, frame_w, frame_h):
    """
    When no patient hand is detected, show the ghost at center-screen
    with a default size.
    """
    target_relative = keyframe["landmarks_relative"]
    target_scale = keyframe["hand_scale"]

    # Default: center of screen, scale to ~30% of frame height
    default_wrist = (frame_w // 2, frame_h // 2)
    default_scale = frame_h * 0.15

    return project_ghost(
        [(r[0] * frame_w, r[1] * frame_h) for r in target_relative],
        target_scale * frame_w,
        default_wrist,
        default_scale
    )


def get_ghost_on_patient(keyframe, patient_points, frame_w, frame_h):
    """
    Project the ghost onto the patient's hand position and scale.
    """
    target_relative_norm = keyframe["landmarks_relative"]
    target_scale = keyframe["hand_scale"]

    # Convert target relative from normalized to pixel space
    target_relative_px = [(r[0] * frame_w, r[1] * frame_h) for r in target_relative_norm]
    target_scale_px = target_scale * frame_w

    # Patient's wrist and scale in pixels
    patient_wrist = patient_points[0]
    patient_scale_px = math.dist(patient_points[0], patient_points[9])
    if patient_scale_px < 1e-6:
        patient_scale_px = frame_h * 0.15

    return project_ghost(target_relative_px, target_scale_px, patient_wrist, patient_scale_px)

# ─── Main ──────────────────────────────────────────────────────────────

def main():
    # Select exercise
    if len(sys.argv) > 1:
        exercise_name = sys.argv[1]
    else:
        exercise_name = select_exercise()

    exercise = load_exercise(exercise_name)
    keyframes = exercise["keyframes"]
    total_steps = len(keyframes)

    if total_steps == 0:
        print("[ERROR] Exercise has no keyframes.")
        sys.exit(1)

    # State
    current_step = 0
    state = "no_hand"      # no_hand, waiting, holding, matched, complete
    match_frac = 0.0
    per_joint_matched = None
    hold_start_time = None
    matched_flash_time = None

    def reset_progress():
        """Reset all playback state back to the very first step."""
        nonlocal current_step, state, match_frac
        nonlocal per_joint_matched, hold_start_time, matched_flash_time
        current_step = 0
        state = "no_hand"
        match_frac = 0.0
        per_joint_matched = None
        hold_start_time = None
        matched_flash_time = None

    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Slightly darken background for contrast
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            now = time.time()
            target_kf = keyframes[min(current_step, total_steps - 1)]

            # ── Process patient's hand ──

            hand_detected = False
            patient_points = None

            if latest_result is not None and latest_result.hand_landmarks:
                for hand_lm in latest_result.hand_landmarks:
                    hand_detected = True
                    patient_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

                    # Compute match
                    patient_angles = compute_angles(patient_points)
                    target_angles = target_kf["angles"]
                    match_frac, per_joint_matched = compute_match_score(
                        patient_angles, target_angles, MATCH_THRESHOLD
                    )

                    break  # only first hand

            # ── State Machine ──

            if state == "complete":
                # Stay in complete state, draw completion screen
                pass

            elif not hand_detected:
                state = "no_hand"
                match_frac = 0.0
                per_joint_matched = None
                hold_start_time = None

            elif match_frac >= MATCH_REQUIRED:
                if state != "holding":
                    state = "holding"
                    hold_start_time = now
                elif (now - hold_start_time) >= HOLD_DURATION:
                    # Advance!
                    current_step += 1
                    if current_step >= total_steps:
                        state = "complete"
                    else:
                        state = "matched"
                        matched_flash_time = now
                    hold_start_time = None
            else:
                state = "waiting"
                hold_start_time = None

            # Brief flash on step advance
            if state == "matched" and matched_flash_time:
                if (now - matched_flash_time) > 0.3:
                    state = "waiting"
                    matched_flash_time = None

            # ── Draw Ghost Hand ──

            if state != "complete":
                if hand_detected and patient_points:
                    ghost_pts = get_ghost_on_patient(target_kf, patient_points, w, h)
                else:
                    ghost_pts = get_ghost_at_default_position(target_kf, w, h)

                draw_ghost_hand(frame, ghost_pts)

            # ── Draw Patient Hand ──

            if hand_detected and patient_points:
                draw_patient_hand(frame, patient_points, per_joint_matched)

            # ── Draw HUD ──

            if state == "complete":
                draw_completion_screen(frame)
            else:
                draw_hud(frame, exercise_name, current_step, total_steps, state, match_frac)

            cv2.imshow("Exercise Guide", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                reset_progress()
                print("[INFO] Exercise restarted.")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
