######## Webcam Object Detection Using Tensorflow-trained Classifier #########
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import imutils
import requests
import importlib.util
from tflite_runtime.interpreter import Interpreter
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import FPS
from pyimagesearch.trackableobject import TrackableObject
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.65)
parser.add_argument('--video', help='Name of the video file',
                    default='')

args = parser.parse_args()
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
API_ENDPOINT = "https://storoc.live/api"
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
if VIDEO_NAME == '':
    VIDEO_PATH = 0
else:
    VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("image width:",imW, "image height:", imH)

previous_occupancy_sent = 0

ct = CentroidTracker()
trackers = []
trackableObjects = {}

total_frames = 0
total_in = 0
total_out = 0
fps = FPS().start()
(W, H) = (None, None)
while (video.isOpened()):
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    rects = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((labels[int(classes[i])] == 'person')):
            print('detected person with confidence:',scores[i])
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH))); xmin = int(max(1, (boxes[i][1] * imW))); ymax = int(min(imH, (boxes[i][2] * imH))); xmax = int(min(imW, (boxes[i][3] * imW)))
                rects.append((xmin, ymin, xmax, ymax))
            # print("rects: ", rects)
            # print("type rects: ", type(rects))
                midpoint = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            # cv2.circle(frame, midpoint, 5, (75, 13, 180), -1)

            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text
    objects = ct.update(rects)
    # print("objects: ", objects)
    # print("type objects: ", type(objects))
    # cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            print("Traveling in direction:", direction)
            print("Centroid[1]:", centroid[1])
            if not to.counted:
                #if centroid[1] > height // 2:#direction > 0: # and centroid[1] < height // 2:
                #if direction > 0 and centroid[1] > 250:
                if centroid[1] > 250:
                    total_out += 1
                    to.counted = True
                #elif centroid[1] < height // 2:# direction < 0: # and centroid[1] > height // 2:
                #elif direction < 0 and centroid[1] < 250:
                elif centroid[1] < 250:
                    total_in += 1
                    to.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # All the results have been drawn on the frame, so it's time to display it.
    info = [
        ("Total In", total_in),
        ("Total Out", total_out),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('Object detector', frame)
    fps.update()
    occupancy = total_in - total_out
    if occupancy != previous_occupancy_sent and occupancy >= 0:
        data = {'unique_id': 'ChIJ82TJ8MaxPIgRGd8xSBhWo54', 'current_occupancy': occupancy}
        requests.post(url=API_ENDPOINT, json=data)
        previous_occupancy_sent = occupancy
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Clean up
video.release()
cv2.destroyAllWindows()
