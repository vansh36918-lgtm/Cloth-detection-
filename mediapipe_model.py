"""
MediaPipe Model Implementation

Concrete implementation of MediaPipe model for human pose estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, List, Optional
import logging

from .models import BaseModel, ModelConfig, ExternalModelWrapper
from ..models.schemas import PoseDetectionResult, KeyPoint, BoundingBox

logger = logging.getLogger(__name__)


class MediaPipeModel(BaseModel):
    """MediaPipe model for human pose estimation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mp_pose = mp.solutions.pose
        self.pose_detector = None
        
        # MediaPipe pose landmark names
        self.landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]
    
    async def _load_model(self) -> None:
        """Load MediaPipe pose model"""
        try:
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=self.config.confidence_threshold,
                min_tracking_confidence=0.5
            )
            self.model = self.pose_detector
            
        except Exception as e:
            self.logger.error(f"Failed to load MediaPipe model: {e}")
            raise
    
    async def preprocess(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess image for MediaPipe"""
        # MediaPipe expects RGB format
        if len(input_data.shape) == 3 and input_data.shape[2] == 3:
            # Convert BGR to RGB
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        
        # Ensure correct data type
        if input_data.dtype != np.uint8:
            input_data = (input_data * 255).astype(np.uint8)
        
        return input_data
    
    async def forward(self, model_input: np.ndarray) -> Any:
        """Run MediaPipe pose detection"""
        results = self.pose_detector.process(model_input)
        return results
    
    async def postprocess(self, model_output: Any) -> Optional[PoseDetectionResult]:
        """Postprocess MediaPipe results"""
        try:
            if not model_output.pose_landmarks:
                return None
            
            landmarks = model_output.pose_landmarks.landmark
            keypoints = []
            
            # Extract keypoints
            for idx, landmark in enumerate(landmarks):
                keypoint = KeyPoint(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z if hasattr(landmark, 'z') else 0.0,
                    visibility=landmark.visibility,
                    name=self.landmark_names[idx] if idx < len(self.landmark_names) else f"keypoint_{idx}"
                )
                keypoints.append(keypoint)
            
            # Calculate bounding box from visible keypoints
            bbox = self._calculate_bbox(keypoints)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(keypoints)
            
            # Estimate body measurements
            measurements = self._estimate_measurements(keypoints)
            
            # Assess pose quality
            pose_quality = self._assess_pose_quality(keypoints)
            
            return PoseDetectionResult(
                keypoints=keypoints,
                bbox=bbox,
                confidence=confidence,
                body_measurements=measurements,
                pose_quality=pose_quality
            )
            
        except Exception as e:
            self.logger.error(f"MediaPipe postprocessing error: {e}")
            return None
    
    def _calculate_bbox(self, keypoints: List[KeyPoint]) -> BoundingBox:
        """Calculate bounding box from keypoints"""
        visible_keypoints = [kp for kp in keypoints if kp.visibility > 0.5]
        
        if not visible_keypoints:
            return BoundingBox(x1=0, y1=0, x2=1, y2=1, confidence=0.0)
        
        xs = [kp.x for kp in visible_keypoints]
        ys = [kp.y for kp in visible_keypoints]
        
        # Add small padding
        padding = 0.05
        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(1, max(xs) + padding)
        y2 = min(1, max(ys) + padding)
        
        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=1.0
        )
    
    def _calculate_confidence(self, keypoints: List[KeyPoint]) -> float:
        """Calculate overall pose confidence"""
        if not keypoints:
            return 0.0
        
        # Use average visibility of key body parts
        key_points_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # shoulders, elbows, wrists, hips, knees
        key_visibilities = [
            keypoints[i].visibility for i in key_points_indices 
            if i < len(keypoints) and keypoints[i].visibility > 0.5
        ]
        
        if not key_visibilities:
            return np.mean([kp.visibility for kp in keypoints])
        
        return np.mean(key_visibilities)
    
    def _estimate_measurements(self, keypoints: List[KeyPoint]) -> Dict[str, float]:
        """Estimate body measurements from keypoints"""
        measurements = {
            "shoulder_width": 0.0,
            "torso_length": 0.0,
            "arm_length": 0.0,
            "leg_length": 0.0
        }
        
        try:
            # Use distance between hips as a secondary reference if needed, 
            # but usually shoulders are best for garments.
            
            # Shoulder width (distance between shoulders)
            if len(keypoints) > 12:
                ls, rs = keypoints[11], keypoints[12]
                if ls.visibility > 0.5 and rs.visibility > 0.5:
                    # Euclidean distance in normalized space
                    measurements["shoulder_width"] = np.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
            
            # Torso length (mid-shoulder to mid-hip or avg shoulder to avg hip)
            if len(keypoints) > 24:
                ls, rs = keypoints[11], keypoints[12]
                lh, rh = keypoints[23], keypoints[24]
                if all(kp.visibility > 0.5 for kp in [ls, rs, lh, rh]):
                    mid_shoulder_y = (ls.y + rs.y) / 2
                    mid_hip_y = (lh.y + rh.y) / 2
                    measurements["torso_length"] = abs(mid_hip_y - mid_shoulder_y)
            
            # Arm length (shoulder to wrist)
            if len(keypoints) > 16:
                ls, lw = keypoints[11], keypoints[15]
                if ls.visibility > 0.5 and lw.visibility > 0.5:
                    measurements["arm_length"] = np.sqrt((ls.x - lw.x)**2 + (ls.y - lw.y)**2)
            
            # Leg length (hip to ankle)
            if len(keypoints) > 28:
                lh, la = keypoints[23], keypoints[27]
                if lh.visibility > 0.5 and la.visibility > 0.5:
                    measurements["leg_length"] = np.sqrt((lh.x - la.x)**2 + (lh.y - la.y)**2)
            
        except Exception as e:
            self.logger.warning(f"Error estimating measurements: {e}")
        
        return measurements
    
    def _assess_pose_quality(self, keypoints: List[KeyPoint]) -> str:
        """Assess the quality of the detected pose"""
        try:
            # Count visible key points
            key_points_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # nose, upper body, legs
            visible_key_points = sum(
                1 for i in key_points_indices 
                if i < len(keypoints) and keypoints[i].visibility > 0.5
            )
            
            total_key_points = len(key_points_indices)
            visibility_ratio = visible_key_points / total_key_points
            
            if visibility_ratio >= 0.8:
                return "good"
            elif visibility_ratio >= 0.6:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _create_dummy_input(self) -> np.ndarray:
        """Create dummy input for warmup"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    async def cleanup(self):
        """Cleanup MediaPipe resources"""
        try:
            if self.pose_detector:
                self.pose_detector.close()
            await super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during MediaPipe cleanup: {e}")


# Factory function for creating MediaPipe model
def create_mediapipe_pose_model(config: ModelConfig) -> MediaPipeModel:
    """Factory function to create MediaPipe pose model"""
    return MediaPipeModel(config)