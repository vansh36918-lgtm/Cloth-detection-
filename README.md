# AR Try-On Model - Module A

This module handles mannequin/garment detection, pose estimation, and measurement calculation using OpenCV and MediaPipe.

## Project Structure
- `mannequin/detect_pose.py`: Pose detection and torso bounding box calculation.
- `mannequin/measure_garment.py`: Physical measurement estimation (cm calibration).
- `mannequin/smoothing.py`: Temporal smoothing for stable measurements.
- `mannequin/test_live.py`: Main entry point for live webcam inference.
- `requirements.txt`: Python dependencies.

## How to Start

### 1. Setup Virtual Environment
Open your terminal in the `ar-tryon-model` folder and run:

```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# OR Activate it (Linux/macOS)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the System
Make sure your webcam is connected and not being used by another app.
```bash
python mannequin/test_live.py
```

## Controls
- **'q'**: Quit the application.
- **'s'**: Save the current cropped garment area to a `.jpg` file.

## Calibration Note
The system uses a standard mannequin shoulder width of **44 cm** for calibration. For the most accurate results, ensure the mannequin is facing the camera directly and is at a reasonable distance.
