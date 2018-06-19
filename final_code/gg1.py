import cv2
import time
from keras.models import load_model
import numpy as np


# Import required libraries for this section

import matplotlib.pyplot as plt
import math
from PIL import Image

# Load facial landmark detector model
model = load_model('my_model.h5')



def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # try to get the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    # Keep video stream open
    while rval:
        # Plot image from camera with detections marked


        # Convert the RGB  image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect the faces in image
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)


        # Get the bounding box for each detected face
        for (x,y,w,h) in faces:
            # Add a red bounding box to the detections image
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)

        cv2.imshow("face detection activated", frame)

        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # exit by pressing any key
            # Destroy windows
            cv2.destroyAllWindows()

            for i in range (1,5):
                cv2.waitKey(1)
            return

        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()


# Run sunglasses painter
laptop_camera_go()
