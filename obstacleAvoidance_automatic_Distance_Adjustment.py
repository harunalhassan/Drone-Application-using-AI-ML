import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from djitellopy import Tello  # Library to control Tello drone
from time import sleep


# Initialize Pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((480, 480))


def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


def getKeyboardInput(distance):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey("a"):
        lr = -speed
    elif getKey("d"):
        lr = speed

    if getKey("w") and distance > 20:  # Prevent moving forward if an obstacle is too close
        fb = speed
    elif getKey("s"):
        fb = -speed

    if getKey("UP"):
        ud = speed
    elif getKey("DOWN"):
        ud = -speed

    if getKey("LEFT"):
        yv = -speed
    elif getKey("RIGHT"):
        yv = speed

    if getKey("e"):
        tello.takeoff()
        sleep(3)

    if getKey("l"):
        tello.land()
        sleep(3)

    return [lr, fb, ud, yv]


# Initialize Pygame
init()

# Initialize Tello drone
tello = Tello()
tello.connect()
tello.streamon()  # Start video stream

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the pre-trained YOLOv8 model

# Constants for distance calculation
KNOWN_DISTANCE = 50.0  # cm (Adjust based on a reference object)
KNOWN_WIDTH = 20.0  # cm (Average object width; adjust for specific objects)


def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width


def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None


# Get initial frame to estimate focal length
frame = tello.get_frame_read().frame
if frame is None:
    print("Error: Could not read frame.")
    tello.streamoff()
    tello.end()
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
    frame = tello.get_frame_read().frame
    if frame is None:
        continue

    # Run YOLO for object detection
    results = model(frame)
    objects = results[0].boxes
    distance = 100  # Default large distance

    for box in objects.xyxy:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        box_width = x2 - x1  # Object width in pixels

        # Estimate distance
        distance = estimate_distance(focal_length, KNOWN_WIDTH, box_width)

        if distance:
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if distance < 20:
                print("WARNING: Object is very close!")
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Automatically move backward if object is too close
                fb_speed = min(50, (20 - distance) * 2)  # Adjust backward speed dynamically
                tello.send_rc_control(0, -fb_speed, 0, 0)
                sleep(0.1)

        # Draw bounding box around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get keyboard inputs and send control signals
    vals = getKeyboardInput(distance)
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    cv2.imshow('Tello Obstacle Detection & Distance Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
