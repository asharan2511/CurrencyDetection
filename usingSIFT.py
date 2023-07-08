import pygame
import pygame.camera
from playsound import playsound
import cv2
from utils import *
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import time

print(cv2.__version__)
print("This is the version")

# Image acquisition from webcam
max_val = 8
max_pt = -1
max_kp = 0

# Initialize SIFT object
sift = cv2.xfeatures2d.SIFT_create()
pygame.camera.init()

# make the list of all available cameras
camlist = pygame.camera.list_cameras()
fimage = 1
# if camera is detected or not
if camlist:
    # initializing the cam variable with default camera
    cam = pygame.camera.Camera(camlist[0], (640, 480))

    # opening the camera
    cam.start()

    # capturing the single image
    image = cam.get_image()
    fimage = image
    # saving the image
    pygame.image.save(image, "test.jpg")
    # width, height = 800, 600
    # screen = pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Image Display")

    # Load the captured image
    image = pygame.image.load("test.jpg")

# Display the image on the screen
# screen.blit(image, (0, 0))
# pygame.display.flip()
# Import the captured image into this program
test_img = cv2.imread("test.jpg")
(kp1, des1) = sift.detectAndCompute(test_img, None)
"""'gray1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
img_1 = cv2.drawKeypoints(gray1, kp1, test_img)
plt.imshow(img_1)
plt.show()
time.sleep(10)"""
# Declaring the training set
training_set = list(paths.list_images(r"C:\Users\shara\Desktop\CurrencyDetection\data"))
flag = 0
for i in range(0, len(training_set)):
    # Train image
    train_img = cv2.imread(training_set[i])
    (kp2, des2) = sift.detectAndCompute(train_img, None)
    if flag == 0:
        gray1 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        img_1 = cv2.drawKeypoints(gray1, kp2, train_img)
        plt.imshow(img_1)
        plt.show()
        time.sleep(10)
        flag = 1
    # Brute force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2
        print(i, " ", training_set[i], " ", len(good))

    if max_val != 8:
        print(training_set[max_pt])

print("Good matches:", max_val)
note = str(training_set[max_pt]).split("\\")[-2]
note = note.strip()
print(note + "\n")
print("\nDetected denomination: Rs.", note)
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(str(note))
screen.blit(fimage, (0, 0))
pygame.display.flip()
playsound(str(note) + ".wav")
