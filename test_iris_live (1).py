"""
test_iris_live.py

-----------------------------------------------------------------
PURPOSE
-----------------------------------------------------------------
Full pipeline combining:
  1. IrisCalibrator  – live iris detection for automatic cm/px scaling
  2. MannequinPoseDetector – body pose landmarks from MediaPipe Tasks
  3. Live measurement display with dynamically calibrated cm values

The key insight:
  Instead of ASSUMING shoulder = 44 cm (hardcoded calibration),
  we measure YOUR iris → know exactly how many cm 1 pixel equals
  at the current camera distance → scale all pose measurements with
  that ratio. The iris width is a fixed biological constant (1.17 cm)
  so it works as a natural ruler in every frame.

-----------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------
  cd d:/AR/ar-tryon-model
  .\\venv\\Scripts\\python.exe mannequin/test_iris_live.py

-----------------------------------------------------------------
IMPORTANT
-----------------------------------------------------------------
You MUST stand close enough to the camera so your eye is clearly
visible for the iris calibration to work. Once "IRIS CALIBRATED"
appears in the top-right corner, you can step back and the ratio
will remain from the last successful reading.

Press 'q' to quit.
"""

import cv2
import math

# Our project modules
from detect_pose       import MannequinPoseDetector
from iris_calibrate    import IrisCalibrator
from smoothing         import MeasurementSmoother


# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

CAMERA_SOURCE = 0     # 0 = default webcam. Change for DroidCam.

# Thresholds for realistic calibration (discard outliers)
MIN_CM_PER_PX = 0.005
MAX_CM_PER_PX = 0.20


# -----------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------

def px_distance(p1, p2):
    """Euclidean distance between two (x, y) pixel points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def measure_from_landmarks(landmarks, frame_w, frame_h, cm_per_px):
    """
    Returns raw measurements in cm using the provided scale ratio.
    """
    if not landmarks or len(landmarks) < 25:
        return None

    l_sh, r_sh = landmarks[11], landmarks[12]
    l_hp, r_hp = landmarks[23], landmarks[24]

    confidence = (l_sh.visibility + r_sh.visibility +
                  l_hp.visibility + r_hp.visibility) / 4.0

    if confidence < 0.3:
        return None

    W, H = frame_w, frame_h
    ls_px = (l_sh.x * W, l_sh.y * H)
    rs_px = (r_sh.x * W, r_sh.y * H)
    lh_px = (l_hp.x * W, l_hp.y * H)
    rh_px = (r_hp.x * W, r_hp.y * H)

    sh_px = px_distance(ls_px, rs_px)
    hp_px = px_distance(lh_px, rh_px)
    
    sh_mid = ((ls_px[0] + rs_px[0]) / 2, (ls_px[1] + rs_px[1]) / 2)
    hp_mid = ((lh_px[0] + rh_px[0]) / 2, (lh_px[1] + rh_px[1]) / 2)
    len_px = px_distance(sh_mid, hp_mid)

    return {
        "shoulderWidth":  sh_px  * cm_per_px,
        "chestWidth":     ((sh_px + hp_px) / 2.0) * cm_per_px,
        "garmentLength":  len_px * cm_per_px,
        "hip_width_cm":   hp_px  * cm_per_px,
        "confidence":     confidence,
    }


def draw_hud(frame, measurements, iris_calibrated, locked):
    """
    Render HUD with stable, smoothed values.
    """
    h, w = frame.shape[:2]

    # --- Calibration status ---
    if locked:
        badge_text, badge_color = "CALIBRATION LOCKED", (0, 180, 255) # orange-blue
    elif iris_calibrated:
        badge_text, badge_color = "IRIS CALIBRATED (Live)", (0, 220, 80) # green
    else:
        badge_text, badge_color = "AWAITING IRIS...", (0, 100, 255) # orange

    cv2.putText(frame, badge_text, (w - 300, 30, ),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, badge_color, 1, cv2.LINE_AA)
    
    cv2.putText(frame, "Press 'l' to Lock/Unlock Scale", (w - 300, 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if not measurements:
        return

    lines = [
        f"Shoulder : {measurements['shoulderWidth']:.1f} cm",
        f"Chest    : {measurements['chestWidth']:.1f} cm",
        f"Hip      : {measurements['hip_width_cm']:.1f} cm",
        f"Length   : {measurements['garmentLength']:.1f} cm",
        f"Confidence : {measurements['confidence']*100:.0f}%",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 180), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 45 + i * 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 80), 1, cv2.LINE_AA)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def main():
    print("-" * 30)
    print("AR Try-On | Stabilized Mode")
    print("-" * 30)
    print("l = Lock/Unlock Calibration")
    print("q = Quit")

    detector   = MannequinPoseDetector()
    calibrator = IrisCalibrator()
    smoother   = MeasurementSmoother(history_size=15)

    current_cm_per_px = None
    cal_locked        = False

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]

        # 1. Update scale only if not locked
        if not cal_locked:
            ratio = calibrator.get_scale_ratio(frame, w, h)
            if ratio:
                current_cm_per_px = ratio

        # 2. Extract pose
        landmarks, annotated_frame, _ = detector.detect_pose(frame)

        # 3. Measure + Smooth
        display_vals = None
        if current_cm_per_px:
            raw = measure_from_landmarks(landmarks, w, h, current_cm_per_px)
            display_vals = smoother.update(raw) if raw else None

        # 4. Display
        draw_hud(annotated_frame, display_vals, (current_cm_per_px is not None), cal_locked)
        cv2.imshow("AR Try-On | Stabilized", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('l'):
            if current_cm_per_px:
                cal_locked = not cal_locked
                print(f"[INFO] Calibration locked: {cal_locked}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
