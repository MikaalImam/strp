"""
Exercise Recording Tool
========================
The physiotherapist performs a hand exercise in front of the camera.
The system captures keyframes of hand poses and saves them as a JSON file.

Controls:
  R     - Start / Stop recording
  S     - Save the recorded exercise
  C     - Clear current recording
  Q     - Quit

Keyframes are captured at ~3 FPS during recording. Only frames where the
hand pose changes significantly are stored, keeping the exercise compact.
"""

import cv2
import mediapipe as mp
import time
import math
import json
import os
from datetime import datetime
from mediapipe.tasks.python import vision

from exercise_utils import (
    HAND_CONNECTIONS, ANGLE_JOINTS,
    calculate_angle, compute_angles, normalize_hand_pose
)

# ─── Config ────────────────────────────────────────────────────────────

MODEL_PATH = "../models/pose_landmarker_lite.task"
SAVE_DIR = "../recorded_exercises"
CAPTURE_INTERVAL = 0.01        # seconds between keyframes
MIN_ANGLE_CHANGE = 2.0         # min degrees change to record a new keyframe

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

# ─── State ─────────────────────────────────────────────────────────────

recording = False
keyframes = []
last_capture_time = 0
last_angles = None
record_start_time = 0

# ─── Helpers ───────────────────────────────────────────────────────────

def pose_changed_enough(new_angles, old_angles):
    """Check if the hand pose changed significantly from last keyframe."""
    if old_angles is None:
        return True
    max_diff = max(abs(n - o) for n, o in zip(new_angles, old_angles))
    return max_diff > MIN_ANGLE_CHANGE


def save_exercise(name):
    """Save recorded keyframes to a JSON file."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_DIR, f"{name}.json")

    data = {
        "name": name,
        "recorded_at": datetime.now().isoformat(),
        "num_keyframes": len(keyframes),
        "keyframes": keyframes
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVED] Exercise '{name}' → {filepath} ({len(keyframes)} keyframes)")
    return filepath


def extract_points(hand_landmarks, w, h):
    """Convert MediaPipe landmarks to pixel (x, y) and normalized (nx, ny) lists."""
    pixel_pts = []
    norm_pts = []
    for lm in hand_landmarks:
        pixel_pts.append((int(lm.x * w), int(lm.y * h)))
        norm_pts.append((lm.x, lm.y))
    return pixel_pts, norm_pts

# ─── Drawing ───────────────────────────────────────────────────────────

def draw_hand(frame, points):
    """Draw the hand skeleton on frame."""
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 200, 0), 2, cv2.LINE_AA)
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_ui(frame, recording, num_keyframes, elapsed):
    """Draw recording UI overlay."""
    h, w = frame.shape[:2]

    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(frame, "EXERCISE RECORDER", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

    # Status
    if recording:
        # Blinking red dot
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (w - 30, 25), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC  {elapsed:.1f}s  [{num_keyframes} frames]",
                    (w - 250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        status = f"{num_keyframes} keyframes recorded" if num_keyframes > 0 else "Ready"
        cv2.putText(frame, status, (w - 280, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
    controls = "[R] Record   [S] Save   [C] Clear   [Q] Quit"
    cv2.putText(frame, controls, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Main Loop ─────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        now = time.time()
        elapsed = now - record_start_time if recording else 0

        # Process detected hand
        if latest_result is not None and latest_result.hand_landmarks:
            for hand in latest_result.hand_landmarks:
                pixel_pts, norm_pts = extract_points(hand, w, h)
                draw_hand(frame, pixel_pts)

                # Capture keyframe if recording
                if recording and (now - last_capture_time) >= CAPTURE_INTERVAL:
                    angles = compute_angles(pixel_pts)

                    if pose_changed_enough(angles, last_angles):
                        relative, scale = normalize_hand_pose(norm_pts)
                        keyframes.append({
                            "time": round(elapsed, 3),
                            "landmarks": norm_pts,
                            "landmarks_relative": relative,
                            "hand_scale": scale,
                            "angles": angles
                        })
                        last_angles = angles
                        last_capture_time = now

        # Draw UI
        draw_ui(frame, recording, len(keyframes), elapsed)
        cv2.imshow("Exercise Recorder", frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = not recording
            if recording:
                record_start_time = time.time()
                last_capture_time = 0
                last_angles = None
                print("[INFO] Recording started...")
            else:
                print(f"[INFO] Recording stopped. {len(keyframes)} keyframes captured.")

        elif key == ord("s"):
            if len(keyframes) == 0:
                print("[WARN] Nothing to save — record an exercise first.")
            else:
                recording = False
                # Use timestamp as default name
                name = input("Enter exercise name (or press Enter for default): ").strip()
                if not name:
                    name = f"exercise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_exercise(name)

        elif key == ord("c"):
            keyframes.clear()
            last_angles = None
            recording = False
            print("[INFO] Recording cleared.")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
