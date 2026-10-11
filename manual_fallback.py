"""
manual_fallback.py

-----------------------------------------------------------------
PURPOSE
-----------------------------------------------------------------
This script is a MANUAL FALLBACK for the AR Try-On system.
It is used when MediaPipe Pose cannot detect a mannequin
(e.g. headless/armless mannequins, poor lighting, partial view).

Instead of AI detection, YOU click 4 key points on the
live camera feed with your mouse, and the script calculates
all garment measurements from those points.

-----------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------
1. Make sure your virtual environment is active:
      .\\venv\\Scripts\\Activate.ps1

2. Run the script:
      python mannequin/manual_fallback.py

-----------------------------------------------------------------
MOUSE CONTROLS (click in this order!)
-----------------------------------------------------------------
  Click 1  →  Left Shoulder
  Click 2  →  Right Shoulder
  Click 3  →  Left Hip
  Click 4  →  Right Hip

-----------------------------------------------------------------
KEYBOARD CONTROLS
-----------------------------------------------------------------
  r   →  Reset   (clear all points and start over)
  s   →  Save    (save the cropped garment area as a .jpg)
  q   →  Quit    (close the application)

-----------------------------------------------------------------
CALIBRATION NOTE
-----------------------------------------------------------------
The pixel-to-cm conversion assumes that the LEFT SHOULDER to
RIGHT SHOULDER distance equals exactly 44 cm (standard
mannequin size). If your subject has different measurements,
update the STANDARD_SHOULDER_CM constant below.
-----------------------------------------------------------------
"""

import cv2
import math

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

# Camera index. 0 = default webcam. Change to 1 or 2 for DroidCam USB.
# For DroidCam Wi-Fi, use: "http://YOUR_PHONE_IP:4747/video"
CAMERA_SOURCE = 0

# Calibration: how many cm does the shoulder-to-shoulder distance equal?
STANDARD_SHOULDER_CM = 44.0

# Padding added around the torso bounding box (10% on each side)
TORSO_PADDING = 0.10

# Labels shown to the user in order, one per click
POINT_LABELS = [
    "1: Left Shoulder",
    "2: Right Shoulder",
    "3: Left Hip",
    "4: Right Hip",
]

# Colours used in drawing (BGR format)
COLOUR_POINT    = (0,   120, 255)     # orange
COLOUR_BOX      = (0,   220,  80)     # green
COLOUR_TEXT_OK  = (0,   220,  80)     # green
COLOUR_TEXT_INS = (255, 255, 255)     # white
COLOUR_HUD_BG   = (20,   20,  20)     # near-black

# -----------------------------------------------------------------
# GLOBALS – shared between mouse callback and main loop
# -----------------------------------------------------------------
points = []       # list of (x, y) tuples, max 4
saved_count = 0   # how many times the user has pressed 's'


# -----------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------

def euclidean(p1, p2):
    """Return the straight-line pixel distance between two (x,y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def midpoint(p1, p2):
    """Return the midpoint of two (x,y) points."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def calc_measurements():
    """
    Given exactly 4 clicked points in order:
        P0 = Left Shoulder
        P1 = Right Shoulder
        P2 = Left Hip
        P3 = Right Hip
    Returns a dict with shoulderWidth, chestWidth, garmentLength (all in cm).
    Returns None if fewer than 4 points are set.
    """
    if len(points) < 4:
        return None

    p_ls, p_rs, p_lh, p_rh = points[0], points[1], points[2], points[3]

    # --- Pixel measurements ---
    shoulder_px = euclidean(p_ls, p_rs)      # shoulder width in pixels
    hip_px      = euclidean(p_lh, p_rh)      # hip width in pixels

    # Garment length = distance between shoulder midpoint and hip midpoint
    shoulder_mid = midpoint(p_ls, p_rs)
    hip_mid      = midpoint(p_lh, p_rh)
    length_px    = euclidean(shoulder_mid, hip_mid)

    # Guard against zero-division (all 4 points in same spot)
    if shoulder_px < 1:
        return None

    # --- Pixel → cm conversion ---
    # We KNOW shoulder_px pixels == STANDARD_SHOULDER_CM cm
    px_to_cm = STANDARD_SHOULDER_CM / shoulder_px

    shoulder_cm = STANDARD_SHOULDER_CM                   # always 44 by definition
    hip_cm      = hip_px    * px_to_cm
    length_cm   = length_px * px_to_cm

    # Chest ≈ average of shoulder and hip widths (better approximation than 1.1× shoulder)
    chest_cm = (shoulder_cm + hip_cm) / 2.0

    return {
        "shoulderWidth":  shoulder_cm,
        "chestWidth":     chest_cm,
        "garmentLength":  length_cm,
        "hip_width_cm":   hip_cm,
    }


def calc_torso_box(frame_shape):
    """
    Calculate a padded bounding box around the 4 clicked points.
    Returns (x_min, y_min, x_max, y_max) clipped to frame boundaries.
    Returns None if fewer than 4 points.
    """
    if len(points) < 4:
        return None

    h, w = frame_shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add padding
    pad_x = int((x_max - x_min) * TORSO_PADDING)
    pad_y = int((y_max - y_min) * TORSO_PADDING)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w - 1, x_max + pad_x)
    y_max = min(h - 1, y_max + pad_y)

    return (x_min, y_min, x_max, y_max)


# -----------------------------------------------------------------
# MOUSE CALLBACK
# -----------------------------------------------------------------

def on_mouse_click(event, x, y, flags, param):
    """
    Called automatically by OpenCV whenever the user interacts with the window.
    We only care about left-button clicks, and only record up to 4 points.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            label = POINT_LABELS[len(points) - 1]
            print(f"[CLICK] {label} at ({x}, {y})")


# -----------------------------------------------------------------
# DRAWING HELPERS
# -----------------------------------------------------------------

def draw_instructions(frame):
    """
    Draw a semi-transparent dark panel with instructions at the top.
    """
    overlay  = frame.copy()
    panel_h  = 50
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), COLOUR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if len(points) < 4:
        # Tell the user what to click next
        next_label = POINT_LABELS[len(points)]
        instruction = f"Click  {next_label}   |   r=reset   s=save   q=quit"
    else:
        instruction = "All 4 points set!   r=reset   s=save garment crop   q=quit"

    cv2.putText(frame, instruction, (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOUR_TEXT_INS, 1, cv2.LINE_AA)


def draw_points(frame):
    """Draw circles and labels at each clicked point."""
    for i, (x, y) in enumerate(points):
        # Filled circle
        cv2.circle(frame, (x, y), 8, COLOUR_POINT, -1)
        # Thin white border for visibility on any background
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)
        # Label next to the circle
        short_label = POINT_LABELS[i].split(": ")[1]   # e.g. "Left Shoulder"
        cv2.putText(frame, short_label, (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_TEXT_INS, 1, cv2.LINE_AA)


def draw_lines(frame):
    """
    After all 4 points are set, draw connecting lines between
    the shoulders and hips to make the shape visible.
    """
    if len(points) < 4:
        return
    p_ls, p_rs, p_lh, p_rh = points
    # Shoulder line and hip line
    cv2.line(frame, p_ls, p_rs, COLOUR_POINT, 2)
    cv2.line(frame, p_lh, p_rh, COLOUR_POINT, 2)
    # Side lines
    cv2.line(frame, p_ls, p_lh, COLOUR_POINT, 2)
    cv2.line(frame, p_rs, p_rh, COLOUR_POINT, 2)


def draw_torso_box(frame, box):
    """Draw the padded garment bounding box in green."""
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_BOX, 2)


def draw_measurements(frame, measurements):
    """
    Draw a semi-transparent HUD panel in the bottom-left
    showing all measurement values.
    """
    if not measurements:
        return

    lines = [
        f"Shoulder : {measurements['shoulderWidth']:.1f} cm",
        f"Chest    : {measurements['chestWidth']:.1f} cm",
        f"Hip      : {measurements['hip_width_cm']:.1f} cm",
        f"Length   : {measurements['garmentLength']:.1f} cm",
    ]

    h = frame.shape[0]
    panel_h  = len(lines) * 30 + 20
    panel_y0 = h - panel_h - 10

    # Semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, panel_y0), (270, h - 10), COLOUR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw each measurement line
    for i, line in enumerate(lines):
        y = panel_y0 + 28 + i * 28
        cv2.putText(frame, line, (20, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOUR_TEXT_OK, 1, cv2.LINE_AA)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def main():
    global saved_count

    # Open camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {CAMERA_SOURCE}")
        print("  → Try changing CAMERA_SOURCE to 1, 2, or a DroidCam URL.")
        return

    # Prefer 720p for better accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "AR Try-On | Manual Fallback"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)   # register our click handler

    print("\n" + "=" * 55)
    print("  AR Try-On – Manual Fallback Mode")
    print("=" * 55)
    print("  Click 4 points in order:")
    for label in POINT_LABELS:
        print(f"    {label}")
    print("\n  r = reset   s = save crop   q = quit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting.")
            break

        # --- Compute derived data ---
        box          = calc_torso_box(frame.shape)
        measurements = calc_measurements()

        # --- Draw all visual layers in order ---
        draw_instructions(frame)   # top instruction bar
        draw_lines(frame)          # connecting lines between points
        draw_points(frame)         # circles for each clicked point
        draw_torso_box(frame, box) # green bounding box
        draw_measurements(frame, measurements)  # HUD panel

        cv2.imshow(window_name, frame)

        # --- Keyboard input ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the application
            print("[INFO] Quitting.")
            break

        elif key == ord('r'):
            # Reset - clear all points and start fresh
            points.clear()
            print("[INFO] Points reset. Start clicking again.")

        elif key == ord('s'):
            # Save the garment crop if a bounding box exists
            if box is not None:
                x1, y1, x2, y2 = box
                # Re-read a fresh frame without drawings for the saved crop
                ret2, raw_frame = cap.read()
                if ret2:
                    crop = raw_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        filename = f"manual_garment_crop_{saved_count}.jpg"
                        cv2.imwrite(filename, crop)
                        print(f"[INFO] Saved: {filename}")
                        saved_count += 1
                    else:
                        print("[WARN] Crop region is empty. Try again.")
                else:
                    print("[WARN] Could not grab a clean frame to save.")
            else:
                print("[WARN] No bounding box yet. Click all 4 points first.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
