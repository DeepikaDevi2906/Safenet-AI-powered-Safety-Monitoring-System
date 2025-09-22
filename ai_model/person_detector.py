import cv2
import numpy as np
from ultralytics import YOLO
from .gender_predictor import predict_gender

# Load YOLO model (for person detection)
yolo_model = YOLO("yolov8m.pt")  # Make sure yolov8m.pt is in the root or provide full path

def process_frame(frame_bytes, location="Unknown"):
    """
    Process a single camera frame:
    - Detect persons using YOLO
    - Crop person regions
    - Predict gender
    - Return True if an alert condition is met
    """
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    alert_generated = False

    # YOLO detection
    results = yolo_model(frame, conf=0.3)[0]  # confidence threshold 0.3

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            # Predict gender for the detected person
            gender = predict_gender(person_crop)

            # Example anomaly: Female detected triggers alert
            if gender == "Female":
                alert_generated = True

    return alert_generated
