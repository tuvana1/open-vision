"""Lane and vehicle detection using OpenCV and YOLOv8."""

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


def detect_lanes(frame):
    """Detect lane markings using Canny + Hough transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI: bottom 40% where lanes appear
    h, w = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi = np.array([[(0, h), (0, int(h * 0.6)), (w, int(h * 0.6)), (w, h)]], np.int32)
    cv2.fillPoly(mask, roi, 255)
    edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            count += 1
    
    return frame, count


def detect_vehicles(frame):
    """Detect vehicles using YOLOv8."""
    # Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    
    count = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        
        if conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            count += 1
    
    return frame, count


def process_video(input_path, output_path):
    """Process video file with lane and vehicle detection."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing: {w}x{h} @ {fps}fps, {total} frames")
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_num = 0
    vehicle_total = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        frame, _ = detect_lanes(frame)
        frame, vehicles = detect_vehicles(frame)
        vehicle_total += vehicles
        
        # TODO: Add configurable overlay
        cv2.putText(frame, f'Frame: {frame_num}/{total} | Vehicles: {vehicles}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Progress: {frame_num/total*100:.1f}%")
    
    cap.release()
    out.release()
    
    # TODO: Return structured stats object
    print(f"\nDone: {frame_num} frames, {vehicle_total} vehicles detected")
    print(f"Output: {output_path}")
    
    return {'frames': frame_num, 'vehicles': vehicle_total, 'fps': fps}
