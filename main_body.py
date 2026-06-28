from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
import base64
import json
from body_detector import body_detector
import asyncio
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static_body", StaticFiles(directory="static_body"), name="static_body")
app.mount("/exercises", StaticFiles(directory="exercises"), name="exercises")

# Initialize hand detector
detector = body_detector()


def get_shoulder_abduction_video_path():
    video_path = Path("exercises") / "shoulder_abduction_with_angles.mp4"
    if video_path.exists() and video_path.is_file():
        return str(video_path)
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
            results, frame_rgb = detector.detect_body(frame)
            
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
