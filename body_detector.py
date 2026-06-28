import os

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class body_detector:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.frame_count = 0
        self.min_landmark_visibility = 0.5

        self.POSE_CONNECTIONS = [
            # Face
            (0, 1),(1, 2),(2, 3),(0, 4),(4, 5),(5, 6),(9, 10),
            
            # Torso
            (11, 12),(11, 23),(12, 24),(23, 24),
            
            # Left arm
            (11, 13),(13, 15),(15, 17),(15, 19),(15, 21),(17, 19),
            
            # Right arm
            (12, 14),(14, 16),(16, 18),(16, 20),(16, 22),(18, 20),
            
            # Left leg
            (23, 25),(25, 27),(27, 29),(27, 31),(29, 31),
            
            # Right leg
            (24, 26),(26, 28),(28, 30),(28, 32),(30, 32),
            ]

    def detect_body(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(self.frame_count * 33)  # ~30 FPS
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        self.frame_count += 1
        return results, frame_rgb
    
    def draw_body_landmarks(self, frame, results):
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape
            connections_show = [(11, 13),(11, 23),(12, 14),(12, 24)]

            for connection in connections_show:
                start_idx, end_idx = connection
                if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                    continue

                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]

                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))

                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)


                cv2.circle(frame, start_point, 5, (0, 0, 255), -1)
                cv2.circle(frame, end_point, 5, (0, 0, 255), -1)

        return frame
    
    def calc_angle_esh_R(self, frame, results):
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape
            
            # Get relevant landmarks
            left_shoulder = pose_landmarks[11]
            left_elbow = pose_landmarks[13]
            left_hip = pose_landmarks[23]

            if not self._landmarks_are_visible(left_shoulder, left_elbow, left_hip):
                return None
            
            # Convert to pixel coordinates
            shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            elbow_point = (int(left_elbow.x * w), int(left_elbow.y * h))
            hip_point = (int(left_hip.x * w), int(left_hip.y * h))
            
            # Calculate vectors as numpy arrays
            shoulder_to_elbow = np.array(elbow_point) - np.array(shoulder_point)
            shoulder_to_hip = np.array(hip_point) - np.array(shoulder_point)

            # Calculate angle using dot product
            dot_product = float(np.dot(shoulder_to_elbow, shoulder_to_hip))

            se_length = float(np.linalg.norm(shoulder_to_elbow))
            sh_length = float(np.linalg.norm(shoulder_to_hip))
            
            if se_length > 0 and sh_length > 0:
                cos_theta = dot_product / (se_length * sh_length)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_rad = math.acos(cos_theta)
                angle_deg = math.degrees(angle_rad)
                return angle_deg
            
        return None
    
    def calc_angle_esh_L(self, frame, results):
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape
            
            # Get relevant landmarks
            right_shoulder = pose_landmarks[12]
            right_elbow = pose_landmarks[14]
            right_hip = pose_landmarks[24]

            if not self._landmarks_are_visible(right_shoulder, right_elbow, right_hip):
                return None
            
            # Convert to pixel coordinates
            shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            elbow_point = (int(right_elbow.x * w), int(right_elbow.y * h))
            hip_point = (int(right_hip.x * w), int(right_hip.y * h))
            
            # Calculate vectors as numpy arrays
            shoulder_to_elbow = np.array(elbow_point) - np.array(shoulder_point)
            shoulder_to_hip = np.array(hip_point) - np.array(shoulder_point)

            # Calculate angle using dot product
            dot_product = float(np.dot(shoulder_to_elbow, shoulder_to_hip))

            se_length = float(np.linalg.norm(shoulder_to_elbow))
            sh_length = float(np.linalg.norm(shoulder_to_hip))
            
            if se_length > 0 and sh_length > 0:
                cos_theta = dot_product / (se_length * sh_length)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_rad = math.acos(cos_theta)
                angle_deg = math.degrees(angle_rad)
                return angle_deg
            
        return None


    def _landmarks_are_visible(self, *landmarks):
        for landmark in landmarks:
            visibility = getattr(landmark, "visibility", 1.0)
            presence = getattr(landmark, "presence", 1.0)

            if visibility < self.min_landmark_visibility:
                return False

            if presence < self.min_landmark_visibility:
                return False

        return True