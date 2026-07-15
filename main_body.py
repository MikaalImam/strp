from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import cv2
import base64
import json
from body_detector import body_detector
import asyncio
from pathlib import Path
import time
import re
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

from zains_code_changed.exercise_utils import (
    POSE_CONNECTIONS,
    compute_angles,
    compute_match_score,
    normalize_pose_landmarks,
    project_ghost,
)

app = FastAPI()

# Mount static files
app.mount("/static_body", StaticFiles(directory="static_body"), name="static_body")
app.mount("/exercises", StaticFiles(directory="exercises"), name="exercises")

# Initialize pose detector
detector = body_detector()

POSE_MODEL_PATH = Path("models") / "pose_landmarker_lite.task"
POSE_EXERCISES_DIR = Path("recorded_exercises")

RECORD_CAPTURE_INTERVAL = 0.33
RECORD_MIN_ANGLE_CHANGE = 5.0

FOLLOW_MATCH_THRESHOLD = 20.0
FOLLOW_MATCH_REQUIRED = 0.75
FOLLOW_HOLD_DURATION = 0.5


def pose_model_exists():
    return POSE_MODEL_PATH.exists() and POSE_MODEL_PATH.is_file()


def get_shoulder_abduction_video_path():
    video_path = Path("exercises") / "shoulder_abduction_with_angles.mp4"
    if video_path.exists() and video_path.is_file():
        return str(video_path)
    return None


def list_recorded_exercises():
    if not POSE_EXERCISES_DIR.exists() or not POSE_EXERCISES_DIR.is_dir():
        return []
    return sorted([f.stem for f in POSE_EXERCISES_DIR.glob("*.json")])


def load_recorded_exercise(name):
    filepath = POSE_EXERCISES_DIR / f"{name}.json"
    if not filepath.exists() or not filepath.is_file():
        return None
    with filepath.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_exercise_name(name):
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    return clean.strip("_")


def encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buffer).decode()


def draw_pose(frame, points):
    for start, end in POSE_CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        cv2.line(frame, points[start], points[end], (255, 200, 0), 2, cv2.LINE_AA)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)


def create_pose_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose_points(landmarker, frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None, None

    pose = result.pose_landmarks[0]
    pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
    norm_pts = [(lm.x, lm.y) for lm in pose]
    return pixel_pts, norm_pts


def pose_changed_enough(new_angles, old_angles):
    if old_angles is None:
        return True
    max_diff = max(abs(n - o) for n, o in zip(new_angles, old_angles))
    return max_diff > RECORD_MIN_ANGLE_CHANGE


def draw_record_overlay(frame, recording, keyframe_count, elapsed_seconds):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.putText(frame, "POSE EXERCISE RECORDER", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

    if recording:
        cv2.putText(frame, f"REC {elapsed_seconds:.1f}s", (w - 190, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{keyframe_count} keyframes", (w - 190, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (w - 28, 30), 9, (0, 0, 255), -1)
    else:
        label = f"{keyframe_count} keyframes saved in memory" if keyframe_count > 0 else "Ready"
        cv2.putText(frame, label, (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 34), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Use buttons below: start/stop recording, save, clear", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def get_ghost_on_patient(keyframe, patient_points, frame_w, frame_h):
    target_relative_norm = keyframe["landmarks_relative"]
    target_scale = keyframe.get("pose_scale", keyframe.get("hand_scale", 1.0))
    target_relative_px = [(r[0] * frame_w, r[1] * frame_h) for r in target_relative_norm]
    target_scale_px = target_scale * frame_w

    patient_anchor = (
        int((patient_points[11][0] + patient_points[12][0]) / 2),
        int((patient_points[11][1] + patient_points[12][1]) / 2),
    )
    patient_scale_px = ((patient_points[12][0] - patient_points[11][0]) ** 2 + (patient_points[12][1] - patient_points[11][1]) ** 2) ** 0.5
    if patient_scale_px < 1e-6:
        patient_scale_px = frame_h * 0.15
    return project_ghost(target_relative_px, target_scale_px, patient_anchor, patient_scale_px)


def get_ghost_at_default_position(keyframe, frame_w, frame_h):
    target_relative = keyframe["landmarks_relative"]
    target_scale = keyframe.get("pose_scale", keyframe.get("hand_scale", 1.0))
    default_anchor = (frame_w // 2, frame_h // 2)
    default_scale = frame_h * 0.15
    return project_ghost(
        [(r[0] * frame_w, r[1] * frame_h) for r in target_relative],
        target_scale * frame_w,
        default_anchor,
        default_scale,
    )


def draw_ghost_pose(frame, ghost_points):
    overlay = frame.copy()
    for start, end in POSE_CONNECTIONS:
        if start < len(ghost_points) and end < len(ghost_points) and ghost_points[start] and ghost_points[end]:
            cv2.line(overlay, ghost_points[start], ghost_points[end], (180, 130, 40), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    for pt in ghost_points:
        if pt is None:
            continue
        cv2.circle(frame, pt, 5, (200, 150, 50), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_follow_overlay(frame, exercise_name, current_step, total_steps, state, match_frac):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.putText(frame, f"EXERCISE: {exercise_name}", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Step {current_step + 1 if state != 'complete' else total_steps}/{total_steps}", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (20, 55), (w - 20, 64), (40, 40, 40), -1)
    progress = (current_step / total_steps) if total_steps else 0.0
    cv2.rectangle(frame, (20, 55), (20 + int((w - 40) * progress), 64), (0, 215, 255), -1)

    match_text = f"Match: {int(match_frac * 100)}%"
    cv2.putText(frame, match_text, (w - 145, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    if state == "complete":
        msg = "Exercise complete! Press Restart."
        color = (0, 255, 0)
    elif state == "holding":
        msg = "Hold this pose..."
        color = (0, 200, 255)
    elif state == "matched":
        msg = "Matched! Moving to next step..."
        color = (0, 255, 0)
    elif state == "waiting":
        msg = "Match the ghost pose"
        color = (180, 180, 180)
    else:
        msg = "Show your pose to the camera"
        color = (120, 120, 120)

    cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, msg, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


async def receive_json_nowait(websocket):
    try:
        text = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
    except asyncio.TimeoutError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

@app.get("/")
async def get_root():
    return FileResponse("pages/home.html")

@app.get("/measure")
async def get_measure_page():
    return FileResponse("pages/index.html")

@app.get("/exercise-menu")
async def get_exercise_menu():
    return FileResponse("pages/exercise_menu.html")

@app.get("/show-exercise")
async def get_show_exercise_page():
    return FileResponse("pages/show_exercise.html")

@app.get("/test")
async def get_test_page():
    return FileResponse("pages/test.html")

@app.get("/record-pose-exercise")
@app.get("/record-hand-exercise")
async def get_record_pose_exercise_page():
    return FileResponse("pages/record_exercise.html")


@app.get("/follow-pose-exercise")
@app.get("/follow-hand-exercise")
async def get_follow_pose_exercise_page():
    return FileResponse("pages/follow_exercise.html")

@app.get("/record-mimic")
async def get_record_mimic_page():
    return FileResponse("pages/record_mimic.html")


@app.get("/api/pose-exercises")
@app.get("/api/hand-exercises")
async def get_pose_exercises():
    return JSONResponse({"exercises": list_recorded_exercises()})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open webcam"})
        await websocket.close()
        return
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            h, w, c = frame.shape
            
            # Detect body
            results, _ = detector.detect_body(frame)
            
            # Draw landmarks
            frame = detector.draw_body_landmarks(frame, results)
            
            # Calculate angle
            angle_right = detector.calc_angle_esh_R(frame, results)
            angle_left = detector.calc_angle_esh_L(frame, results)


            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode()
            
            data = {
                "frame": frame_base64,
                "angle_right": float(angle_right) if angle_right is not None else None,
                "angle_left": float(angle_left) if angle_left is not None else None
            }            

            await websocket.send_json(data)
            
            # Small delay to control frame rate
            await asyncio.sleep(0.01)
    
    except WebSocketDisconnect:
        cap.release()
    except Exception as e:
        print(f"Error: {e}")
        cap.release()
        await websocket.close()


@app.websocket("/ws/pose/record")
@app.websocket("/ws/hand/record")
async def pose_record_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not pose_model_exists():
        await websocket.send_json({"error": f"Missing pose model at {POSE_MODEL_PATH}"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open webcam"})
        await websocket.close()
        return

    landmarker = create_pose_landmarker()
    recording = False
    keyframes = []
    last_capture_time = 0.0
    last_angles = None
    record_start_time = 0.0
    event_message = None

    recording = False
    countdown_active = False
    countdown_start_time = 0.0
    COUNTDOWN_DURATION = 5

    try:
        while True:
            success, frame = cap.read()
            if not success:
                await websocket.send_json({"error": "Failed to read webcam frame"})
                break

            frame = cv2.flip(frame, 1)
            now = time.time()

            if countdown_active:
                countdown_remaining = max(
                    0,
                    math.ceil(COUNTDOWN_DURATION - (now - countdown_start_time))
                )

                if countdown_remaining == 0:
                    countdown_active = False
                    recording = True
                    record_start_time = now
                    last_capture_time = 0.0
                    last_angles = None
                    event_message = "Recording started."
            else:
                countdown_remaining = None

            elapsed = now - record_start_time if recording else 0.0

            pixel_pts, norm_pts = detect_pose_points(landmarker, frame)
            if pixel_pts:
                draw_pose(frame, pixel_pts)

                if recording and (now - last_capture_time) >= RECORD_CAPTURE_INTERVAL:
                    angles = compute_angles(pixel_pts)
                    if pose_changed_enough(angles, last_angles):
                        relative, scale = normalize_pose_landmarks(norm_pts)
                        keyframes.append(
                            {
                                "time": round(elapsed, 3),
                                "landmarks": norm_pts,
                                "landmarks_relative": relative,
                                "pose_scale": scale,
                                "angles": angles,
                            }
                        )
                        last_angles = angles
                        last_capture_time = now

            draw_record_overlay(frame, recording, len(keyframes), elapsed)
            if countdown_active and countdown_remaining is not None:
                countdown_text = str(countdown_remaining)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 4
                thickness = 8

                (text_width, text_height), _ = cv2.getTextSize(
                    countdown_text,
                    font,
                    font_scale,
                    thickness
                )

                frame_height, frame_width = frame.shape[:2]

                text_position = (
                    (frame_width - text_width) // 2,
                    (frame_height + text_height) // 2
                )

                # Black outline
                cv2.putText(
                    frame,
                    countdown_text,
                    text_position,
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 6,
                    cv2.LINE_AA
                )

                # White number
                cv2.putText(
                    frame,
                    countdown_text,
                    text_position,
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
                
            frame_base64 = encode_frame(frame)
            if frame_base64 is None:
                await websocket.send_json({"error": "Failed to encode frame"})
                break

            payload = {
                "frame": frame_base64,
                "recording": recording,
                "countdown": countdown_remaining,
                "num_keyframes": len(keyframes),
                "elapsed": round(elapsed, 3),
            }
            if event_message:
                payload["event"] = event_message
                event_message = None
            await websocket.send_json(payload)

            command = await receive_json_nowait(websocket)
            if command:
                action = command.get("action")
                if action == "start_recording":
                    recording = False
                    countdown_active = True
                    countdown_start_time = time.time()
                    last_capture_time = 0.0
                    last_angles = None
                    event_message = "Recording will begin in 5 seconds."
                
                elif action == "stop_recording":
                    recording = False
                    countdown_active = False
                    event_message = f"Recording stopped. {len(keyframes)} keyframes captured."

                elif action == "clear_recording":
                    recording = False
                    countdown_active = False
                    keyframes.clear()
                    last_angles = None
                    event_message = "Recording cleared."

                elif action == "save_recording":
                    if not keyframes:
                        event_message = "Nothing to save. Record an exercise first."
                    else:
                        raw_name = command.get("name") or f"exercise_{int(time.time())}"
                        exercise_name = sanitize_exercise_name(raw_name)
                        if not exercise_name:
                            exercise_name = f"exercise_{int(time.time())}"

                        POSE_EXERCISES_DIR.mkdir(parents=True, exist_ok=True)
                        filepath = POSE_EXERCISES_DIR / f"{exercise_name}.json"
                        data = {
                            "name": exercise_name,
                            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "num_keyframes": len(keyframes),
                            "keyframes": keyframes,
                        }
                        with filepath.open("w", encoding="utf-8") as handle:
                            json.dump(data, handle, indent=2)
                        event_message = f"Saved as {exercise_name}.json"

            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        landmarker.close()


@app.websocket("/ws/pose/follow/{exercise_name}")
@app.websocket("/ws/hand/follow/{exercise_name}")
async def pose_follow_websocket_endpoint(websocket: WebSocket, exercise_name: str):
    await websocket.accept()
    if not pose_model_exists():
        await websocket.send_json({"error": f"Missing pose model at {POSE_MODEL_PATH}"})
        await websocket.close()
        return

    exercise = load_recorded_exercise(exercise_name)
    if exercise is None:
        await websocket.send_json({"error": f"Exercise '{exercise_name}' not found"})
        await websocket.close()
        return

    keyframes = exercise.get("keyframes") or []
    total_steps = len(keyframes)
    if total_steps == 0:
        await websocket.send_json({"error": "Exercise has no keyframes"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open webcam"})
        await websocket.close()
        return

    landmarker = create_pose_landmarker()
    current_step = 0
    state = "no_pose"
    match_frac = 0.0
    hold_start_time = None
    matched_flash_time = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                await websocket.send_json({"error": "Failed to read webcam frame"})
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)
            now = time.time()

            target_kf = keyframes[min(current_step, total_steps - 1)]
            pose_detected = False
            patient_points = None

            pixel_pts, _ = detect_pose_points(landmarker, frame)
            if pixel_pts:
                pose_detected = True
                patient_points = pixel_pts
                patient_angles = compute_angles(patient_points)
                target_angles = target_kf["angles"]
                match_frac, _ = compute_match_score(patient_angles, target_angles, FOLLOW_MATCH_THRESHOLD)

            if state == "complete":
                pass
            elif not pose_detected:
                state = "no_pose"
                match_frac = 0.0
                hold_start_time = None
            elif match_frac >= FOLLOW_MATCH_REQUIRED:
                if state != "holding":
                    state = "holding"
                    hold_start_time = now
                elif hold_start_time is not None and (now - hold_start_time) >= FOLLOW_HOLD_DURATION:
                    current_step += 1
                    if current_step >= total_steps:
                        current_step = total_steps
                        state = "complete"
                    else:
                        state = "matched"
                        matched_flash_time = now
                    hold_start_time = None
            else:
                state = "waiting"
                hold_start_time = None

            if state == "matched" and matched_flash_time and (now - matched_flash_time) > 0.3:
                state = "waiting"
                matched_flash_time = None

            if state != "complete":
                if patient_points:
                    ghost_points = get_ghost_on_patient(target_kf, patient_points, w, h)
                else:
                    ghost_points = get_ghost_at_default_position(target_kf, w, h)
                draw_ghost_pose(frame, ghost_points)

            if patient_points:
                draw_pose(frame, patient_points)

            draw_follow_overlay(frame, exercise_name, min(current_step, total_steps - 1), total_steps, state, match_frac)

            frame_base64 = encode_frame(frame)
            if frame_base64 is None:
                await websocket.send_json({"error": "Failed to encode frame"})
                break

            await websocket.send_json(
                {
                    "frame": frame_base64,
                    "state": state,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "match_fraction": round(match_frac, 3),
                }
            )

            command = await receive_json_nowait(websocket)
            if command and command.get("action") == "restart":
                current_step = 0
                state = "no_pose"
                match_frac = 0.0
                hold_start_time = None
                matched_flash_time = None

            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        landmarker.close()


@app.websocket("/ws/exercise")
@app.websocket("/ws/exercise/")
@app.websocket("/ws/exercise/shoulder_abduction")
@app.websocket("/ws/exercise/shoulder_abduction/")
async def exercise_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    video_path = get_shoulder_abduction_video_path()
    if video_path is None:
        await websocket.send_json({"error": "shoulder_abduction.mp4 not found in exercises folder"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open exercise video"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps and fps > 0 else 0.03

    try:
        while True:
            ret, frame = cap.read()

            # Restart when the file reaches the end so the preview stays available.
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results, _ = detector.detect_body(frame)
            frame = detector.draw_body_landmarks(frame, results)
            angle_right = detector.calc_angle_esh_R(frame, results)
            angle_left = detector.calc_angle_esh_L(frame, results)

            angle_text_right = f"Right Angle: {angle_right:.1f} deg" if angle_right is not None else "Right Angle: --"
            angle_text_left = f"Left Angle: {angle_left:.1f} deg" if angle_left is not None else "Left Angle: --"

            cv2.putText(
                frame, angle_text_right, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, )

            cv2.putText(
                frame, angle_text_left, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, )

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode()

            await websocket.send_json(
                {
                    "frame": frame_base64,
                    "angle_right": angle_right,
                    "angle_left": angle_left,
                    "video": Path(video_path).name,
                }
            )

            await asyncio.sleep(frame_delay)

    except WebSocketDisconnect:
        cap.release()
    except Exception as e:
        print(f"Error: {e}")
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
