import math

class GarmentMeasurer:
    def __init__(self):
        # Standard mannequin shoulder width used for calibration (in cm)
        # This is only used when no depth info is available.
        self.standard_shoulder_cm = 44.0

    def _distance(self, p1, p2):
        """Euclidean distance between two 2D pixel points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def measure(self, landmarks, frame_width, frame_height):
        """
        Measure garment dimensions from pose landmarks.
        landmarks: list of NormalizedLandmark from MediaPipe Tasks
        Returns a dict with shoulderWidth, chestWidth, garmentLength, confidence (all in cm).
        """
        if not landmarks or len(landmarks) < 25:
            return None

        # ---- Key landmark indices (MediaPipe Pose) ----
        # 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER
        # 23: LEFT_HIP,      24: RIGHT_HIP
        # 13: LEFT_ELBOW,    14: RIGHT_ELBOW
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip      = landmarks[23]
        r_hip      = landmarks[24]

        # ---- Confidence: average visibility of key joints ----
        confidence = (l_shoulder.visibility + r_shoulder.visibility +
                      l_hip.visibility + r_hip.visibility) / 4.0

        # ---- Skip low-confidence frames ----
        if confidence < 0.4:
            return None

        # ---- Convert normalized coords to pixel coords ----
        W, H = frame_width, frame_height

        ls_px = (l_shoulder.x * W, l_shoulder.y * H)
        rs_px = (r_shoulder.x * W, r_shoulder.y * H)
        lh_px = (l_hip.x * W,      l_hip.y * H)
        rh_px = (r_hip.x * W,      r_hip.y * H)

        # ---- Shoulder width in pixels ----
        shoulder_width_px = self._distance(ls_px, rs_px)

        if shoulder_width_px < 5:   # too small – likely bad detection
            return None

        # ---- pixel → cm ratio ----
        # CALIBRATION: shoulder_width_px pixels == 44cm
        px_to_cm = self.standard_shoulder_cm / shoulder_width_px

        # ---- Shoulder width (should always be ~44 cm, sanity check) ----
        shoulder_width_cm = shoulder_width_px * px_to_cm   # always 44

        # ---- Hip width in pixels → chest approximation ----
        hip_width_px   = self._distance(lh_px, rh_px)
        hip_width_cm   = hip_width_px * px_to_cm

        # Chest ≈ mid-point between shoulder and hip width
        # Garment/chest width realistically is between shoulder and hip
        chest_width_cm = (shoulder_width_cm + hip_width_cm) / 2.0

        # ---- Garment length: shoulder midpoint → hip midpoint ----
        shoulder_mid = ((ls_px[0] + rs_px[0]) / 2, (ls_px[1] + rs_px[1]) / 2)
        hip_mid      = ((lh_px[0] + rh_px[0]) / 2, (lh_px[1] + rh_px[1]) / 2)
        garment_length_px = self._distance(shoulder_mid, hip_mid)
        garment_length_cm = garment_length_px * px_to_cm

        return {
            "shoulderWidth": shoulder_width_cm,
            "chestWidth":    chest_width_cm,
            "garmentLength": garment_length_cm,
            "hip_width_cm":  hip_width_cm,
            "confidence":    confidence,
        }
