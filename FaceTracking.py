import cv2
import numpy as np
import pygame
from djitellopy import Tello
import time

# Initialize Pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((480, 480))

def getKey(keyName):
    ans = False
    for event in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

# Initialize Tello Drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

# Face Tracking Variables
w, h = 480, 480  # Updated resolution
fbRange = [6500, 9000]  # Increased range for better tracking
pid = [0.4, 0.4, 0]  # PID control for smoother movement
pError = 0
takeoff = False  # Ensure drone does not take off immediately

# Face Detection
def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# Face Tracking Logic (Optimized)
def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]

    fb = 0  # Forward/backward movement
    yv = 0  # Yaw (rotation)

    # Calculate error for alignment
    error = x - (w // 2)
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -30, 30))  # Limit rotation speed

    # Adjust forward/backward movement
    if fbRange[0] < area < fbRange[1]:
        fb = 0  # Stay in place
    elif area > fbRange[1]:
        fb = -10  # Move backward slowly if face is too close
    elif 5000 < area < fbRange[0]:
        fb = 10   # Move forward slowly if face is too far

    # Handle rotation
    if x == 0:
        yv = 0  # Stop rotating if no face is detected
        error = 0
    else:
        yv = speed  # Rotate drone based on face position

    tello.send_rc_control(0, fb, 0, yv)
    return error

# Manual Keyboard Control
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    global takeoff

    if getKey("a"):
        lr = -speed
    elif getKey("d"):
        lr = speed

    if getKey("w"):
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

    if getKey("t") and not takeoff:  # Take off when 'T' is pressed
        tello.takeoff()
        time.sleep(3)
        takeoff = True

    if getKey("l"):  # Land when 'L' is pressed
        tello.land()
        time.sleep(3)
        takeoff = False  # Reset takeoff flag

    return [lr, fb, ud, yv]

# Initialize Pygame
init()

while True:
    img = tello.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)

    vals = getKeyboardInput()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    cv2.imshow("Tello Face Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
