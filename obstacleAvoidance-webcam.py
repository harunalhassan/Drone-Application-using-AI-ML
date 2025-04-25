import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the pre-trained YOLOv8 model

# Constants for distance calculation
KNOWN_DISTANCE = 50.0  # cm (Adjust based on a reference object)
KNOWN_WIDTH = 20.0  # cm (Average object width; adjust for specific objects)

# Function to estimate focal length
def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width

# Function to estimate distance from bounding box width
def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None

# Open webcam
cap = cv2.VideoCapture(0)

# Get initial frame to estimate focal length
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Run YOLO once to get an initial object width
results = model(frame)
objects = results[0].boxes

if len(objects) > 0:
    first_box = objects.xyxy[0]  # Get the first detected object's bounding box
    box_width = first_box[2] - first_box[0]  # Calculate width
    focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, box_width)
    print(f"Estimated Focal Length: {focal_length:.2f} cm")
else:
    focal_length = 600  # Default focal length (adjust if no object detected)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO for object detection
    results = model(frame)
    objects = results[0].boxes

    for box in objects.xyxy:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        box_width = x2 - x1  # Object width in pixels

        # Estimate distance
        distance = estimate_distance(focal_length, KNOWN_WIDTH, box_width)

        if distance:
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Warning if object is too close
            if distance < 20:
                print("WARNING: Object is very close!")
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Draw bounding box around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Obstacle Detection & Distance Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
