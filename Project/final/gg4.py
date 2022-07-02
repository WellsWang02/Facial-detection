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

    sunglasses = cv2.imread("iceberg.jpg", cv2.IMREAD_UNCHANGED)

    # Keep video stream open
    while rval:
        # Plot image from camera with detections marked


        # Convert the RGB  image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect the faces in image
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        # predict keypoints
        predicted_points = []




        for (x,y,w,h) in faces:
            crop_img = gray[y:y+h, x:x+w]
            resized_crop_image = cv2.resize(crop_img, (96, 96))
            reshape_img = np.reshape(resized_crop_image, (96,96,1)) / 255

            predicted_points.append(reshape_img)

            # Add a red bounding box to the detections image
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
            #
            # # plot our image and the detected facial points
            # fig = plt.figure(figsize = (9,9))
            # ax1 = fig.add_subplot(111)
            # ax1.set_xticks([])
            # ax1.set_yticks([])
        ar = np.array(predicted_points)
        # print(ar.shape)
        if predicted_points != []:
            predicted_points = model.predict(ar)

            for i in range(predicted_points.shape[0]):
                orig_x,orig_y,orig_w,orig_h = faces[i]

                # denormalize points
                pts_x = predicted_points[i][0::2] * orig_w/2 + orig_w/2 + orig_x
                pts_y = predicted_points[i][1::2] * orig_h/2 + orig_h/2 + orig_y
                # for j in range(len(pts_x)):
                #     cv2.circle(frame, (pts_x[j], pts_y[j]), 3, (0,255,0), -1)
                # frame.scatter(pts_x,pts_y, marker='o', c='lawngreen', s=10)

                sunglasses_height = int((pts_y[10] - pts_y[9])/1.1)
                sunglasses_width = int((pts_x[7] - pts_x[9]) * 1.1)

                sunglasses_top_left_y = int(pts_y[9])
                sunglasses_top_left_x = int(pts_x[9])

                # resized sunglasses
                resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

                # region that is transparent
                alpha_region = resized_sunglasses[:,:,3] != 0

                frame[sunglasses_top_left_y:sunglasses_top_left_y+sunglasses_height, sunglasses_top_left_x:sunglasses_top_left_x+sunglasses_width,:][alpha_region] = resized_sunglasses[:,:,:3][alpha_region]

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

