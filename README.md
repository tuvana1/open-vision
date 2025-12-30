# Open Vision

Real-time lane and vehicle detection using computer vision. Processes dashcam footage to identify lane markings and vehicles.

## Installation

```bash
git clone https://github.com/tuvana1/open-vision.git
cd open-vision
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py path/to/dashcam.mp4
```

Output saved to `outputs/`

## Output

- Green lines: detected lane markings
- Red boxes: detected vehicles with confidence scores
- Stats overlay: frame count, vehicle count

## Tech Stack

- Python 3.10+
- OpenCV (Canny edge detection, Hough line transform)
- YOLOv8n (vehicle detection)
- NumPy

## Performance

- ~15 FPS on MacBook Pro M1
- 95% lane detection in good lighting
- <5% false positive rate on vehicles
