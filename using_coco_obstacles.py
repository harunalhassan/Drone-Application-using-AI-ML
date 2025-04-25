import cv2
import numpy as np
import pygame
import json
import time
from ultralytics import YOLO

# Load COCO Object Widths from JSON
with open("coco_object_widths.json", "r") as file:
    OBJECT_WIDTHS = json.load(file)


# Initialize Pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((400, 400))


def getKey(keyName):
    """Check if a key is pressed"""
    ans = False
    for eve in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


# Function to get real-world object width from COCO dataset
def get_real_width(label):
    """Fetch real-world width of object dynamically"""
    return OBJECT_WIDTHS.get(label, 20)  # Default 20 cm if not found


def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width


def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None


# Initialize Pygame
init()

# Open Webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Get initial frame to estimate focal length
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    cap.release()
    exit()

# Run YOLO once to get an initial object width
results = model(frame)
objects = results[0].boxes

if len(objects) > 0:
    first_box = objects.xyxy[0]  # Get the first detected object's bounding box
    label = results[0].names[int(objects.cls[0])]  # Get class label
    real_width = get_real_width(label)  # Fetch real width from COCO
    box_width = first_box[2] - first_box[0]  # Calculate width in pixels

    focal_length = calculate_focal_length(50, real_width, box_width)  # Known Distance = 50 cm
    print(f"Estimated Focal Length: {focal_length:.2f} cm for {label}")
else:
    focal_length = 600  # Default focal length (adjust if no object detected)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLO for object detection
    results = model(frame)
    objects = results[0].boxes

    for i in range(len(objects)):
        box = objects.xyxy[i]
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        box_width = x2 - x1  # Object width in pixels
        label = results[0].names[int(objects.cls[i])]  # Get class label
        real_width = get_real_width(label)  # Fetch real-world width dynamically

        # Estimate distance
        front_distance = estimate_distance(focal_length, real_width, box_width)

        if front_distance:
            cv2.putText(frame, f"{label}: {front_distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if front_distance < 20:
                print("WARNING: Object is very close!")
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Draw bounding box around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Webcam Object Detection & Distance Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
