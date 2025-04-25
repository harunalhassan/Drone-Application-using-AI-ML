import cv2
from djitellopy import tello
import pygame
from time import sleep, time

# Initialize pygame for keyboard control
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

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey("LEFT"):
        lr = -speed
    elif getKey("RIGHT"):
        lr = speed

    if getKey("UP"):
        fb = speed
    elif getKey("DOWN"):
        fb = -speed

    if getKey("w"):
        ud = speed
    elif getKey("s"):
        ud = -speed

    if getKey("a"):
        yv = -speed
    elif getKey("d"):
        yv = speed

    if getKey("q"):
        me.land()
        sleep(3)

    if getKey("e"):
        me.takeoff()

    if getKey("z"):
        captureImage()

    return [lr, fb, ud, yv]

def captureImage():
    cv2.imwrite(f"Capture_{int(time())}.jpg", img)
    print("Image Captured!")

# Initialize pygame
init()

# Initialize the drone
me = tello.Tello()
me.connect()
print(f"Battery: {me.get_battery()}%")
me.streamoff()
me.streamon()

while True:
    # Get keyboard inputs
    vals = getKeyboardInput()

    # Ensure the drone stays in place if no key is pressed
    if vals == [0, 0, 0, 0]:
        me.send_rc_control(0, 0, 0, 0)
    else:
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the drone's camera feed
    img = me.get_frame_read().frame

    # Display the video feed
    cv2.imshow("Drone Feed", img)
    cv2.waitKey(1)
