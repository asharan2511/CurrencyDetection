import os

# from picamera import PiCamera
from time import sleep
import cv2
import sys
import utils
from utils import *

# Importing the necessary library functions
import subprocess
import numpy as np
from imutils import paths

# Image acquisition using RasPi camera
max_val = 8
max_pt = -1
max_kp = 0
orb = cv2.ORB_create()
# Importing the captured image into this program
test_img = cv2.imread("testfiles/test8.jpg")
# display('original', original)
(kp1, des1) = orb.detectAndCompute(test_img, None)
# Declaring the training set

training_set = list(
    paths.list_images(
        r"C:\Users\shara\Desktop\Mini project\indian-currency-classification-master\data"
    )
)

for i in range(0, len(training_set)):
    # train image
    train_img = cv2.imread(training_set[i])
    (kp2, des2) = orb.detectAndCompute(train_img, None)
    # brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    # if good then append to list of good matches
    for m, n in all_matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2
        print(i, " ", training_set[i], " ", len(good))
    if max_val != 8:
        print(training_set[max_pt])
print("good matches", max_val)
note = str(training_set[max_pt])[13:-6][6:]
print("\nDetected denomination: Rs. ", note)
# os.system('aplay/home/pi/Desktop/project/'+str(note)+'.wav')
