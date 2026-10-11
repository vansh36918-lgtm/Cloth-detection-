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

        p_right = (int(lm[self.IRIS_RIGHT_EDGE].x * W),
                   int(lm[self.IRIS_RIGHT_EDGE].y * H))
        p_left  = (int(lm[self.IRIS_LEFT_EDGE].x  * W),
                   int(lm[self.IRIS_LEFT_EDGE].y  * H))

        iris_width_px = self._dist(p_right, p_left)
        if iris_width_px < 1:
            return None

        cm_per_pixel = self.IRIS_DIAMETER_CM / iris_width_px

        # --- Stabilization ---
        self.ratio_history.append(cm_per_pixel)
        
        # Only return the average if we have enough samples for stability
        if len(self.ratio_history) < 10:
            return None
            
        avg_ratio = sum(self.ratio_history) / len(self.ratio_history)

        # --- Visual feedback ---
        cx     = (p_right[0] + p_left[0]) // 2
        cy     = (p_right[1] + p_left[1]) // 2
        radius = max(2, int(iris_width_px / 2))
        cv2.circle(frame, (cx, cy), radius, (0, 220, 80), 2)
        cv2.circle(frame, (cx, cy), 2,      (0, 220, 80), -1)

        return cm_per_pixel
