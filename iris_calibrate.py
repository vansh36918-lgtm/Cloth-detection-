"""
iris_calibrate.py

Uses MediaPipe Tasks FaceLandmarker (Python 3.12 compatible)
to detect iris landmarks and compute the cm_per_pixel scale.

BIOLOGY:  Average adult iris diameter = 1.17 cm (constant).
          cm_per_pixel = 1.17 / iris_width_in_pixels

IRIS INDICES inside the FaceLandmarker output (468 total landmarks):
  Right iris (subject's left eye from camera POV):
    468 – center  |  469 – top  |  470 – right  |  471 – bottom  |  472 – left
  Left iris (subject's right eye from camera POV):
    473 – center  |  474 – top  |  475 – right  |  476 – bottom  |  477 – left

For horizontal iris width we use indices 470 and 472 (right edge / left edge
of the right iris) — most stable across head rotations.

HOW TO USE:
    from iris_calibrate import IrisCalibrator
    calibrator = IrisCalibrator()
    cm_per_px = calibrator.get_scale_ratio(frame, width, height)
"""

import cv2
import math
import os
import mediapipe as mp


class IrisCalibrator:
    """Detects the iris in a live frame and returns the cm_per_pixel ratio."""

    IRIS_DIAMETER_CM = 1.17   # biological constant – average adult iris width

    # Right iris (subject's left eye):  indices 468-472
    # We use right-edge (470) and left-edge (472) for horizontal diameter
    IRIS_RIGHT_EDGE = 470
    IRIS_LEFT_EDGE  = 472

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

        BaseOptions        = mp.tasks.BaseOptions
        FaceLandmarker     = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode        = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Stabilization: history of the last 30 valid scale ratios
        from collections import deque
        self.ratio_history = deque(maxlen=30)

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_scale_ratio(self, frame, frame_width, frame_height):
        """
        Detect iris landmarks in `frame` and return cm_per_pixel.
        Draws a small green circle on the iris as visual feedback.
        Returns float (cm_per_pixel) or None.
        """
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results   = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        # FaceLandmarker returns normalised landmarks (x,y,z in 0..1)
        lm = results.face_landmarks[0]   # first (and only) face

        # Safety: FaceLandmarker with iris outputs 478 landmarks
        if len(lm) < 478:
            return None

        W, H = frame_width, frame_height

        # --- Head Rotation / Face Symmetry Check ---
        # 4 = Nose tip, 468 = Right iris center, 473 = Left iris center
        p_nose = (lm[4].x * W, lm[4].y * H)
        p_right_center = (lm[468].x * W, lm[468].y * H)
        p_left_center = (lm[473].x * W, lm[473].y * H)

        dist_nose_right = self._dist(p_nose, p_right_center)
        dist_nose_left  = self._dist(p_nose, p_left_center)

        if dist_nose_right > 0 and dist_nose_left > 0:
            symmetry_ratio = min(dist_nose_right, dist_nose_left) / max(dist_nose_right, dist_nose_left)
            # If the head is rotated too much, we skip this frame to prevent perspective error
            if symmetry_ratio < 0.90:
                if len(self.ratio_history) >= 10:
                    # Calculate trimmed mean/average from history
                    vals = sorted(self.ratio_history)
                    n = len(vals)
                    if n >= 15:
                        trim = int(n * 0.15)
                        trimmed = vals[trim:n-trim]
                        avg_ratio = sum(trimmed) / len(trimmed)
                    else:
                        avg_ratio = sum(vals) / n
                    focal_length_px = 1.0 * W
                    distance_to_camera_cm = focal_length_px * avg_ratio
                    return {
                        "scale_ratio": avg_ratio,
                        "distance_cm": distance_to_camera_cm
                    }
                return None

        # --- Dual-Iris Width Measurement ---
        # Right iris (subject's left eye): right edge (470), left edge (472)
        p_right_r = (lm[470].x * W, lm[470].y * H)
        p_right_l = (lm[472].x * W, lm[472].y * H)
        width_right = self._dist(p_right_r, p_right_l)

        # Left iris (subject's right eye): right edge (475), left edge (477)
        p_left_r  = (lm[475].x * W, lm[475].y * H)
        p_left_l  = (lm[477].x * W, lm[477].y * H)
        width_left  = self._dist(p_left_r, p_left_l)

        # Average the widths if both are valid to reduce noise
        if width_right > 1 and width_left > 1:
            iris_width_px = (width_right + width_left) / 2.0
        elif width_right > 1:
            iris_width_px = width_right
        elif width_left > 1:
            iris_width_px = width_left
        else:
            return None

        cm_per_pixel = self.IRIS_DIAMETER_CM / iris_width_px

        # --- Stabilization with Outlier Filtering ---
        self.ratio_history.append(cm_per_pixel)
        
        # Only return the average if we have enough samples for stability
        if len(self.ratio_history) < 10:
            return None
            
        # Robust trimmed average (sort and drop top/bottom 15% to eliminate tracking spikes)
        vals = sorted(self.ratio_history)
        n = len(vals)
        if n >= 15:
            trim = int(n * 0.15)
            trimmed = vals[trim:n-trim]
            avg_ratio = sum(trimmed) / len(trimmed)
        else:
            avg_ratio = sum(vals) / n

        # --- Visual feedback (draw circles on both irises) ---
        for cx_px, cy_px, w_px in [( (p_right_r[0]+p_right_l[0])/2, (p_right_r[1]+p_right_l[1])/2, width_right ),
                                   ( (p_left_r[0]+p_left_l[0])/2,   (p_left_r[1]+p_left_l[1])/2,   width_left )]:
            cx, cy = int(cx_px), int(cy_px)
            radius = max(2, int(w_px / 2))
            cv2.circle(frame, (cx, cy), radius, (0, 220, 80), 2)
            cv2.circle(frame, (cx, cy), 2,      (0, 220, 80), -1)

        focal_length_px = 1.0 * W
        distance_to_camera_cm = focal_length_px * avg_ratio

        return {
            "scale_ratio": avg_ratio,
            "distance_cm": distance_to_camera_cm
        }

