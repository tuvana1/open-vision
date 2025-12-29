"""Test detection on a single image."""

import cv2
import os
from detector import detect_lanes, detect_vehicles


def main():
    os.makedirs("outputs", exist_ok=True)
    
    path = 'uploads/test_frame.jpg'
    if not os.path.exists(path):
        print(f"Add image to {path}")
        return
    
    frame = cv2.imread(path)
    if frame is None:
        print("Cannot read image")
        return
    
    frame, lanes = detect_lanes(frame)
    frame, vehicles = detect_vehicles(frame)
    
    print(f"Lanes: {lanes}, Vehicles: {vehicles}")
    cv2.imwrite('outputs/test_result.jpg', frame)


if __name__ == "__main__":
    main()
