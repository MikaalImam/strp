# Hand Distance Measurer - Web App

A real-time hand distance measurement web application built with FastAPI, MediaPipe, and WebSocket.

## Features

✅ Real-time hand detection using MediaPipe  
✅ Measures distance between two hands  
✅ Live video stream in browser  
✅ Clean, responsive UI  
✅ No external dependencies except Python packages  

## Tech Stack

**Backend:**
- FastAPI
- MediaPipe
- OpenCV (cv2)
- WebSocket for real-time communication

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### 3. Open in Browser

Navigate to `http://localhost:8000` in your web browser

## Usage

1. Click the **Start** button to begin capturing video
2. Position both hands in front of the webcam
3. The app will display:
   - Live video feed with hand landmarks
   - A line connecting the two hands
   - Distance measurement in pixels
4. Click **Stop** to end the session

## How It Works

1. **Backend (FastAPI + MediaPipe):**
   - Captures frames from your webcam
   - Detects hands using MediaPipe
   - Draws landmarks on the frame
   - Calculates the distance between hand centers
   - Sends processed frames to the frontend via WebSocket

2. **Frontend (HTML/JS):**
   - Receives video frames as base64-encoded JPEG images
   - Displays them in real-time
   - Shows distance measurements
   - Manages WebSocket connection

## Project Structure

```
mp_test/
├── main.py              # FastAPI server with WebSocket
├── hand_detector.py     # MediaPipe hand detection logic
├── requirements.txt     # Python dependencies
├── index.html          # Main HTML page
└── static/
    ├── app.js          # Frontend WebSocket client
    └── style.css       # Styling
```

## API Endpoints

### WebSocket: `/ws`

**Message Format (Server → Client):**
```json
{
  "frame": "base64_encoded_jpeg_image",
  "distance": 0.35,
  "hand1_center": [100, 150],
  "hand2_center": [250, 180],
  "width": 640,
  "height": 480
}
```

## Requirements

- Python 3.8+
- Webcam
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

**"Cannot open webcam"**
- Check if your webcam is accessible and not in use by another application
- Grant browser permissions to access the webcam

**Low FPS or lag**
- Ensure your system has adequate resources
- Close other applications using the webcam
- Check your internet connection (if running remotely)

**MediaPipe errors**
- Reinstall MediaPipe: `pip install --upgrade mediapipe`

## License

MIT
