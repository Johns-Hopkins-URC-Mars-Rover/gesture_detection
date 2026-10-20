import time
import cv2
import torch
from ultralytics import YOLO

# 1. Configuration
MODEL_PATH = "yolov8s_best.pt"  # Path to your downloaded weights
CAMERA_INDEX = 0               # 0 is typically the built-in webcam; use 1 or 2 for external USB cameras
CONF_THRESHOLD = 0.40          # Minimum detection confidence
IMG_SIZE = 640                 # Inference resolution (match training resolution)

# 2. Select compute device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_PATH} on {device.upper()}...")

model = YOLO(MODEL_PATH)
model.to(device)

# 3. Open webcam stream
cap = cv2.VideoCapture(CAMERA_INDEX)

# Optional: set camera frame buffer / resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"Error: Could not open video device at index {CAMERA_INDEX}")
    exit()

print("Camera started. Press 'q' to exit the stream.")

prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # 4. Run inference
        # stream=True yields results as a memory-efficient generator
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False
        )

        # 5. Extract annotated frame (bounding boxes, class labels, confidences)
        annotated_frame = results[0].plot()

        # 6. Calculate & render FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 7. Display output
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Press 'q' to break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 8. Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed released.")