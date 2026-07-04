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
        
        self.connections_show = [
                # Right arm
                (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),

                # Left arm
                (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),

                # Torso
                (11, 12), (11, 23), (12, 24), (23, 24)
            ]

        # Progressive target zone settings
        self.target_step = 10
        self.target_start_offset = 5
        self.target_zone_size = 15 #the size of the target zone in degrees

        self.final_min_angle = 95
        self.final_max_angle = 105
        self.reset_below_angle = 20

        self.required_hold_frames = 5

        self.arm_targets = {
            "left": {
                "lower": None,
                "upper": None,
                "hold_frames": 0,
                "completed": False
            },
            "right": {
                "lower": None,
                "upper": None,
                "hold_frames": 0,
                "completed": False
            }
        }

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
            # connections_show = [(11, 13),(11, 23),(12, 14),(12, 24), (14, 16), (13, 15)]


            for connection in self.connections_show:
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
        
    def landmark_to_point(self, landmark, w, h):
        return (
            int(landmark.x * w),
            int(landmark.y * h)
        )

    def update_arm_target_zone(self, side, current_angle):
        if current_angle is None:
            return None, None, False

        target = self.arm_targets[side]

        # First time setup
        if target["lower"] is None or target["upper"] is None:
            target["lower"] = current_angle + self.target_start_offset
            target["upper"] = target["lower"] + self.target_zone_size
            target["hold_frames"] = 0
            target["completed"] = False

        # Do not let target go beyond final range
        if target["lower"] >= self.final_min_angle:
            target["lower"] = self.final_min_angle
            target["upper"] = self.final_max_angle

        lower = target["lower"]
        upper = target["upper"]

        # Check if user is inside the current target zone
        if lower <= current_angle <= upper:
            target["hold_frames"] += 1
        else:
            target["hold_frames"] = 0

        # If user stayed in the target zone for enough frames
        if target["hold_frames"] >= self.required_hold_frames:
            target["hold_frames"] = 0

            # If user reached final target, send wedge back down to around 30 degrees
            if lower == self.final_min_angle and upper == self.final_max_angle:
                target["lower"] = self.reset_below_angle
                target["upper"] = self.reset_below_angle + self.target_zone_size
                target["completed"] = False
                return target["lower"], target["upper"], target["completed"]

            # Otherwise move target higher
            target["lower"] += self.target_step
            target["upper"] += self.target_step

            if target["lower"] >= self.final_min_angle:
                target["lower"] = self.final_min_angle
                target["upper"] = self.final_max_angle

        return target["lower"], target["upper"], target["completed"]
    
    def draw_arm_target_zone(self, frame, shoulder_point, arm_length, lower_angle, upper_angle, side):
        if lower_angle is None or upper_angle is None:
            return frame

        overlay = frame.copy()

        points = [shoulder_point]

        for angle in range(int(lower_angle), int(upper_angle) + 1):
            angle_rad = np.deg2rad(angle)

            # Angle is measured from the torso/downward direction.
            # 0 degrees = arm down
            # 90 degrees = arm horizontal
            # 100+ degrees = slightly above horizontal
            if side == "left":
                x = int(shoulder_point[0] - arm_length * np.sin(angle_rad))
            else:
                x = int(shoulder_point[0] + arm_length * np.sin(angle_rad))

            y = int(shoulder_point[1] + arm_length * np.cos(angle_rad))

            points.append((x, y))

        points = np.array(points, dtype=np.int32)

        # Yellow transparent target zone
        cv2.fillPoly(overlay, [points], (0, 255, 255))

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0
        )

        # Draw boundary lines for clarity
        for angle in [lower_angle, upper_angle]:
            angle_rad = np.deg2rad(angle)

            if side == "left":
                x = int(shoulder_point[0] - arm_length * np.sin(angle_rad))
            else:
                x = int(shoulder_point[0] + arm_length * np.sin(angle_rad))

            y = int(shoulder_point[1] + arm_length * np.cos(angle_rad))

            cv2.line(
                frame,
                shoulder_point,
                (x, y),
                (0, 200, 255),
                3
            )

        return frame
    
    def draw_progressive_arm_target(self, frame, results, side):
        if not results.pose_landmarks:
            return frame

        pose_landmarks = results.pose_landmarks[0]
        h, w, _ = frame.shape

        if side == "right":
            shoulder_idx = 11
            elbow_idx = 13
            wrist_idx = 15
            hip_idx = 23
            current_angle = self.calc_angle_esh_R(frame, results)
        else:
            shoulder_idx = 12
            elbow_idx = 14
            wrist_idx = 16
            hip_idx = 24
            current_angle = self.calc_angle_esh_L(frame, results)

        shoulder = pose_landmarks[shoulder_idx]
        elbow = pose_landmarks[elbow_idx]
        wrist = pose_landmarks[wrist_idx]
        hip = pose_landmarks[hip_idx]

        if not self._landmarks_are_visible(shoulder, elbow, wrist, hip):
            return frame

        shoulder_point = self.landmark_to_point(shoulder, w, h)
        wrist_point = self.landmark_to_point(wrist, w, h)

        arm_length = int(
            np.linalg.norm(
                np.array(wrist_point) - np.array(shoulder_point)
            )
        )

        if arm_length <= 0:
            return frame

        lower_angle, upper_angle, completed = self.update_arm_target_zone(
            side,
            current_angle
        )

        frame = self.draw_arm_target_zone(
            frame,
            shoulder_point,
            arm_length,
            lower_angle,
            upper_angle,
            side
        )

        # Text feedback
        # if completed:
        #     text = f"{side.capitalize()} arm: completed"
        #     color = (0, 255, 0)
        # else:
        #     text = f"{side.capitalize()} target: {int(lower_angle)}-{int(upper_angle)} deg"
        #     color = (0, 255, 255)

        # cv2.putText(
        #     frame,
        #     text,
        #     (30, 40 if side == "left" else 80),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     color,
        #     2
        # )

        # if current_angle is not None:
        #     cv2.putText(
        #         frame,
        #         f"{side.capitalize()} angle: {int(current_angle)} deg",
        #         (30, 65 if side == "left" else 105),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.7,
        #         color,
        #         2
        #     )

        return frame
    
    def reset_arm_targets(self):
        self.arm_targets = {
            "left": {
                "lower": None,
                "upper": None,
                "hold_frames": 0,
                "completed": False
            },
            "right": {
                "lower": None,
                "upper": None,
                "hold_frames": 0,
                "completed": False
            }
        }

    def head_center_radius(self, nose, left_ear, right_ear, w, h):
        nose_point = (
            int(nose.x * w),
            int(nose.y * h)
        )

        left_ear_point = (
            int(left_ear.x * w),
            int(left_ear.y * h)
        )

        right_ear_point = (
            int(right_ear.x * w),
            int(right_ear.y * h)
        )

        ear_distance = int(
            np.linalg.norm(
                np.array(left_ear_point) - np.array(right_ear_point)
            )
        )

        # OpenCV ellipse axes are radii, not full width/height
        head_width = max(int(ear_distance * 0.95), 45)
        head_height = max(int(ear_distance * 1.40), 70)

        # Nose is near the center/front of face, so move oval center down slightly
        head_center = (
            nose_point[0],
            nose_point[1] + int(head_height * 0.35)
        )

        return head_center, head_width, head_height
    

    #not feeling as uselfull
    #have removed as of now
    def silhouette(self, frame, results):
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape


            mask = np.zeros((h, w), dtype=np.uint8)

            for start_idx, end_idx in self.connections_show:
                if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                    continue

                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]

                start_point = (
                    int(start_landmark.x * w),
                    int(start_landmark.y * h)
                )

                end_point = (
                    int(end_landmark.x * w),
                    int(end_landmark.y * h)
                )

                cv2.line(mask, start_point, end_point, 255, 45)
                cv2.circle(mask, start_point, 30, 255, -1)
                cv2.circle(mask, end_point, 30, 255, -1)

            try:
                nose = pose_landmarks[0]
                left_ear = pose_landmarks[7]
                right_ear = pose_landmarks[8]

                head_center, head_width, head_height = self.head_center_radius(nose, left_ear, right_ear, w, h)

                cv2.ellipse(mask, head_center, (head_width, head_height), 0, 0, 360, 255, -1)
            except Exception:
                pass

            if not np.any(mask):
                return frame

            soft_mask = cv2.GaussianBlur(
                mask,
                (31, 31),
                0
            ).astype(np.float32) / 255.0

            soft_mask = soft_mask[..., None]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            output = (
                frame.astype(np.float32) * soft_mask +
                gray_frame.astype(np.float32) * (1.0 - soft_mask)
            ).astype(np.uint8)

            return output

        return frame