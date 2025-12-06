# Hand Proximity Alert System

A simple OpenCV-based application that detects a hand and alerts when it approaches a virtual boundary on the right side of the frame. No external libraries are used.
Demo: https://drive.google.com/drive/u/0/folders/10qbbZVBBMi2JF3_csRJ2tpjS_00WwAIa
## Features
- Real-time hand detection using HSV skin segmentation + background subtraction
- Visual boundary and distance measurement
- Smooth state classification: SAFE, WARNING, DANGER
- On-screen FPS and threshold zones
- Interactive controls for calibration and threshold adjustment

## Requirements
- Python 3.8+
- Webcam accessible by the system
- Packages: `opencv-python`, `numpy`

## Installation
You can install dependencies using `uv` (fast) or `pip`.

```powershell
# Using uv (recommended)
uv pip install opencv-python numpy

# Or using pip
pip install opencv-python numpy
```

## Run
From the project folder:

```powershell
python main.py
```

## Controls
- `q`: Quit
- `r`: Reset background model
- `+`: Increase warning/danger thresholds by 10px
- `-`: Decrease warning/danger thresholds by 10px (floors at 20/10)
- `c`: Calibrate skin color range based on center region of current frame

## How It Works
- Resizes frames to 640x480 for speed
- Segments skin tones in HSV using two hue ranges
- Optionally refines foreground with MOG2 background subtraction
- Finds largest contour as the hand and computes extreme points and an approximate palm center
- Measures horizontal distance to a virtual boundary near the right side
- Classifies state based on smoothed distance vs thresholds and overlays visual feedback

## Troubleshooting
- "Could not open camera": Ensure a webcam is connected and not in use by another app; try changing `camera_index` in `main.py` (e.g., `1` or `2`).
- Poor detection or flicker: Adjust room lighting; press `c` to calibrate; tweak thresholds with `+`/`-`.
- High CPU usage: Close other camera apps; reduce resolution or kernel sizes in code if needed.
- Black window or no frames: Confirm permissions to access the camera; update your webcam driver.

## Customization
- Change boundary position: Edit `self.virtual_object_x` in `__init__`.
- Adjust thresholds: Edit `self.warning_threshold` and `self.danger_threshold`.
- Tune skin ranges: Modify `self.lower_skin`/`self.upper_skin` and use calibration (`c`).



