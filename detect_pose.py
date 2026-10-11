import cv2
import mediapipe as mp
import numpy as np
import os

class MannequinPoseDetector:
    def __init__(self):
        # Using modern Tasks API for compatibility with Python 3.12
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Ensure the model file exists
        model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
    def detect_pose(self, frame):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image
        results = self.landmarker.detect(mp_image)
        
        annotated_frame = frame.copy()
        torso_box = None
        pose_landmarks = None
        
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0] # Single person
            
            # Extract relevant points
            h, w, _ = frame.shape
            
            # MediaPipe Tasks output indices are the same as legacy
            # 11: L_SHOULDER, 12: R_SHOULDER, 23: L_HIP, 24: R_HIP
            l_shoulder = pose_landmarks[11]
            r_shoulder = pose_landmarks[12]
            l_hip = pose_landmarks[23]
            r_hip = pose_landmarks[24]
            
            # Calculate coordinates in pixels
            points = [
                (int(l_shoulder.x * w), int(l_shoulder.y * h)),
                (int(r_shoulder.x * w), int(r_shoulder.y * h)),
                (int(l_hip.x * w), int(l_hip.y * h)),
                (int(r_hip.x * w), int(r_hip.y * h))
            ]
            
            # Draw landmarks (simplified version since mp.solutions is missing)
            for p in points:
                cv2.circle(annotated_frame, p, 5, (0, 0, 255), -1)
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min = max(0, min(x_coords))
            y_min = max(0, min(y_coords))
            x_max = min(w, max(x_coords))
            y_max = min(h, max(y_coords))
            
            # Add some padding to the torso box
            padding_x = int((x_max - x_min) * 0.1)
            padding_y = int((y_max - y_min) * 0.1)
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(w, x_max + padding_x)
            y_max = min(h, y_max + padding_y)
            
            if x_min < x_max and y_min < y_max:
                torso_box = (x_min, y_min, x_max, y_max)
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
        return pose_landmarks, annotated_frame, torso_box
