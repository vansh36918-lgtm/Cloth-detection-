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


def measure_from_landmarks(landmarks, frame_w, frame_h, cm_per_px, distance_cm=None):
    """
    Returns raw measurements in cm using the provided scale ratio.
    Applies 3D yaw, 2D roll pose corrections and depth-offset perspective correction.
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

    # --- 1. Pose Yaw and Roll Angle Computations & Corrections ---
    # 2D Shoulder Roll (Roll/Tilt angle in the image plane)
    dx_sh = ls_px[0] - rs_px[0]
    dy_sh = ls_px[1] - rs_px[1]
    roll_angle = math.atan2(dy_sh, dx_sh)
    roll_deg = math.degrees(roll_angle)

    # 3D Shoulder Yaw (Yaw/Rotation angle out of plane)
    dx_sh_norm = l_sh.x - r_sh.x
    dz_sh_norm = l_sh.z - r_sh.z
    yaw_angle = math.atan2(dz_sh_norm, dx_sh_norm)
    yaw_deg = math.degrees(yaw_angle)

    # Safe clamped cosines to avoid dividing by tiny numbers or introducing noise
    cos_roll = max(0.85, math.cos(roll_angle))
    cos_yaw  = max(0.85, math.cos(yaw_angle))

    # Raw pixel distances
    sh_px_raw = px_distance(ls_px, rs_px)
    hp_px_raw = px_distance(lh_px, rh_px)

    # Compensate pixel distances for roll & yaw foreshortening
    sh_px = sh_px_raw / (cos_roll * cos_yaw)
    hp_px = hp_px_raw / (cos_roll * cos_yaw)

    # Vertical roll correction for length
    sh_mid = ((ls_px[0] + rs_px[0]) / 2, (ls_px[1] + rs_px[1]) / 2)
    hp_mid = ((lh_px[0] + rh_px[0]) / 2, (lh_px[1] + rh_px[1]) / 2)
    len_px_raw = px_distance(sh_mid, hp_mid)

    dx_len = sh_mid[0] - hp_mid[0]
    dy_len = sh_mid[1] - hp_mid[1]
    len_roll_angle = math.atan2(dx_len, dy_len)
    cos_len_roll = max(0.85, math.cos(len_roll_angle))
    len_px = len_px_raw / cos_len_roll

    # --- 2. Dynamic Perspective Depth Correction ---
    # Torso is typically ~9.0 cm deeper than the eyes
    TORSO_DEPTH_OFFSET_CM = 9.0
    if distance_cm:
        corrected_cm_per_px = cm_per_px * (1.0 + TORSO_DEPTH_OFFSET_CM / distance_cm)
    else:
        # Fallback default offset scaling (assuming avg distance of 120cm)
        corrected_cm_per_px = cm_per_px * (1.0 + TORSO_DEPTH_OFFSET_CM / 120.0)

    # Collect posture warnings
    warnings = []
    if abs(yaw_deg) > 12.0:
        warnings.append("FACE CAMERA DIRECTLY")
    if abs(roll_deg) > 10.0:
        warnings.append("STAND UPRIGHT (DON'T LEAN)")
    if distance_cm:
        if distance_cm < 80.0:
            warnings.append("STEP BACK SLIGHTLY")
        elif distance_cm > 220.0:
            warnings.append("STEP CLOSER SLIGHTLY")

    return {
        "shoulderWidth":  sh_px  * corrected_cm_per_px,
        "chestWidth":     ((sh_px + hp_px) / 2.0) * corrected_cm_per_px,
        "garmentLength":  len_px * corrected_cm_per_px,
        "hip_width_cm":   hp_px  * corrected_cm_per_px,
        "confidence":     confidence,
        "yaw_deg":        yaw_deg,
        "roll_deg":       roll_deg,
        "warnings":       warnings,
    }


def draw_hud(frame, measurements, iris_calibrated, locked, distance_cm=None):
    """
    Render HUD with stable, smoothed values and posture guidance.
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

    # Draw Distance if available
    if distance_cm:
        dist_text = f"Distance: {distance_cm/100:.2f} m"
        cv2.putText(frame, dist_text, (w - 300, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if not measurements:
        return

    # Draw measurements HUD
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

    # Draw Posture Warnings
    warnings = measurements.get("warnings", [])
    if warnings:
        warn_overlay = frame.copy()
        box_w, box_h = 420, len(warnings) * 25 + 15
        x0, y0 = (w - box_w) // 2, h - box_h - 40
        cv2.rectangle(warn_overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 180), -1)
        cv2.addWeighted(warn_overlay, 0.7, frame, 0.3, 0, frame)
        
        for idx, warning in enumerate(warnings):
            cv2.putText(frame, f"WARNING: {warning}", (x0 + 15, y0 + 20 + idx * 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_guide_overlay(frame):
    """Draw a silhouette guide on the screen to help the user align themselves."""
    h, w = frame.shape[:2]
    # Draw head guide: an oval in the upper center
    head_cx, head_cy = w // 2, int(h * 0.25)
    head_rx, head_ry = int(w * 0.08), int(h * 0.12)
    cv2.ellipse(frame, (head_cx, head_cy), (head_rx, head_ry), 0, 0, 360, (0, 220, 255), 1, cv2.LINE_AA)
    
    # Draw shoulder guide
    sh_y = int(h * 0.40)
    sh_w = int(w * 0.28)
    cv2.line(frame, (head_cx - sh_w // 2, sh_y), (head_cx + sh_w // 2, sh_y), (0, 220, 255), 1, cv2.LINE_AA)
    
    # Draw body guide (dotted box for torso)
    cv2.rectangle(frame, (head_cx - sh_w // 2, sh_y), (head_cx + sh_w // 2, int(h * 0.85)), (0, 220, 255), 1, cv2.LINE_AA)
    
    # Add guide text
    cv2.putText(frame, "Align your body with the guide", (head_cx - 130, int(h * 0.92)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


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

    current_cm_per_px   = None
    current_distance_cm = None
    cal_locked          = False

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]

        # 1. Update scale only if not locked
        if not cal_locked:
            cal_data = calibrator.get_scale_ratio(frame, w, h)
            if cal_data:
                current_cm_per_px   = cal_data["scale_ratio"]
                current_distance_cm = cal_data["distance_cm"]

        # 2. Extract pose
        landmarks, annotated_frame, _ = detector.detect_pose(frame)

        # 3. Measure + Smooth
        display_vals = None
        if current_cm_per_px:
            raw = measure_from_landmarks(landmarks, w, h, current_cm_per_px, current_distance_cm)
            if raw:
                display_vals = smoother.update(raw)
                if display_vals:
                    display_vals["yaw_deg"]  = raw["yaw_deg"]
                    display_vals["roll_deg"] = raw["roll_deg"]
                    display_vals["warnings"] = raw["warnings"]

        # 4. Draw overlays
        if not current_cm_per_px:
            draw_guide_overlay(annotated_frame)
            
        draw_hud(annotated_frame, display_vals, (current_cm_per_px is not None), cal_locked, current_distance_cm)
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
