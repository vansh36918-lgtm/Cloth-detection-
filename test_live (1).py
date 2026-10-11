import cv2
import sys
from detect_pose import MannequinPoseDetector
from measure_garment import GarmentMeasurer
from smoothing import MeasurementSmoother

# ─────────────────────────────────────────────────────────────
#  CAMERA CONFIGURATION
#  DroidCam options:
#    1. USB mode:   cv2.VideoCapture(1)  or  cv2.VideoCapture(2)
#    2. Wi-Fi mode: cv2.VideoCapture("http://192.168.1.100:4747/video")
#
#  To use DroidCam: Update CAMERA_SOURCE below with your phone IP!
# ─────────────────────────────────────────────────────────────
CAMERA_SOURCE = 0                                          # Default to primary webcam
# CAMERA_SOURCE = 1                                          # Try secondary/USB webcam
# CAMERA_SOURCE = "http://192.168.1.100:4747/video"          # Example DroidCam IP


def draw_hud(frame, smoothed, torso_box):
    """Draw translucent HUD panel with measurement data."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the left
    overlay   = frame.copy()
    panel_w   = 280
    panel_h   = 200
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    conf = smoothed.get("confidence", 0)
    bar_color = (0, 220, 80) if conf > 0.75 else (0, 170, 255) if conf > 0.50 else (0, 80, 255)

    lines = [
        (f"Shoulder : {smoothed['shoulderWidth']:.1f} cm",  (0, 220, 80)),
        (f"Chest    : {smoothed['chestWidth']:.1f} cm",     (0, 220, 80)),
        (f"Hip      : {smoothed['hip_width_cm']:.1f} cm",   (0, 220, 80)),
        (f"Length   : {smoothed['garmentLength']:.1f} cm",  (0, 220, 80)),
        (f"Conf     : {conf*100:.0f}%",                     bar_color),
    ]

    for i, (text, color) in enumerate(lines):
        y = 45 + i * 30
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "AR-TryOn  Module A", (20, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def try_open_camera(source, fallback_indices=(0, 1, 2)):
    """Try to open source; if it fails, fallback to first working webcam index."""
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print(f"[INFO] Camera opened: {source}")
        return cap
    print(f"[WARN] Could not open {source}. Trying fallbacks…")
    for idx in fallback_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Fallback camera opened at index {idx}")
            return cap
    return None


def main():
    print("=" * 55)
    print("  AR Try-On – Module A  |  pose + garment measurement")
    print("=" * 55)
    print(f"  Camera source : {CAMERA_SOURCE}")
    print("  Press 'q' to quit | 's' to save garment crop")
    print("=" * 55)

    detector = MannequinPoseDetector()
    measurer  = GarmentMeasurer()
    smoother  = MeasurementSmoother(history_size=20)

    cap = try_open_camera(CAMERA_SOURCE)
    if cap is None:
        print("[ERROR] No camera could be opened. Check DroidCam IP or USB connection.")
        return

    # Prefer higher resolution for better accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Reconnecting…")
            cap.release()
            cap = try_open_camera(CAMERA_SOURCE)
            if cap is None:
                break
            continue

        h, w = frame.shape[:2]

        # ── 1. Detect pose ──────────────────────────────────────
        landmarks, annotated_frame, torso_box = detector.detect_pose(frame)

        garment_crop = None

        # ── 2. Crop garment region ──────────────────────────────
        if torso_box:
            x1, y1, x2, y2 = torso_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 < x2 and y1 < y2:
                garment_crop = frame[y1:y2, x1:x2]
                if garment_crop.size > 0:
                    # Resize for consistent display
                    disp = cv2.resize(garment_crop, (320, 400))
                    cv2.imshow("Garment Crop", disp)

        # ── 3. Measure + smooth ─────────────────────────────────
        measurements = measurer.measure(landmarks, w, h)
        smoothed = smoother.update(measurements) if measurements else None

        # ── 4. Draw HUD ─────────────────────────────────────────
        if smoothed:
            draw_hud(annotated_frame, smoothed, torso_box)
        else:
            cv2.putText(annotated_frame, "Stand in front of camera – full body visible",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)

        cv2.imshow("AR Try-On | Pose Detection", annotated_frame)

        # ── 5. Input handling ───────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if garment_crop is not None and garment_crop.size > 0:
                filename = f"garment_capture_{saved_count}.jpg"
                cv2.imwrite(filename, garment_crop)
                print(f"[INFO] Saved: {filename}")
                saved_count += 1
            else:
                print("[WARN] No garment crop to save yet.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
