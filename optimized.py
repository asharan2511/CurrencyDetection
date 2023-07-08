import pygame
import pygame.camera
from playsound import playsound
import cv2
import numpy as np
from imutils import paths

# Initialize Pygame
pygame.init()

# Initialize Pygame camera
pygame.camera.init()

# Initialize SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Make the list of all available cameras
camlist = pygame.camera.list_cameras()
fimage = None

# Check if camera is available
if camlist:
    # Initialize the camera with default camera
    cam = pygame.camera.Camera(camlist[0], (640, 480))

    # Start the camera
    cam.start()

    # Capture the image
    image = cam.get_image()
    fimage = image

    # Convert the image to OpenCV format
    test_img = pygame.surfarray.array3d(image)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite("test.jpg", test_img)

    # Load the captured image
    image = pygame.image.load("test.jpg")

# Display the image using Pygame
if fimage is not None:
    width, height = fimage.get_width(), fimage.get_height()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Image Display")
    screen.blit(fimage, (0, 0))
    pygame.display.flip()

# Import the captured image into this program
test_img = cv2.imread("test.jpg")
(kp1, des1) = sift.detectAndCompute(test_img, None)

# Declaring the training set
training_set = list(paths.list_images("C:/Users/shara/Desktop/Mini project/indian-currency-classification-master/data"))

max_val = 8
max_pt = -1
max_kp = None

for i in range(len(training_set)):
    # Train image
    train_img = cv2.imread(training_set[i])
    (kp2, des2) = sift.detectAndCompute(train_img, None)

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

if max_val != 8 and max_pt != -1:
    print(training_set[max_pt])

    # Get the recognized denomination
    note = training_set[max_pt].split("\\")[-2].strip()
    print("\nDetected denomination: Rs.", note)

    # Display the recognized denomination using Pygame
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(note)
    screen.blit(fimage, (0, 0))
    pygame.display.flip()

    # Play the corresponding sound
    playsound(note + ".wav")
else:
    print("No match found.")

# Stop the camera
if camlist:
    cam.stop()

# Quit Pygame
pygame.quit()