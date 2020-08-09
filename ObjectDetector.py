import argparse
import os
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
from VideoStream import VideoStream
import time

class ObjectDetector:

    def __init__(self):
        self.setup()

    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            required=True)
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--resolution',
                            help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='1280x720')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args = parser.parse_args()

        self.MODEL_NAME = args.modeldir
        self.GRAPH_NAME = args.graph
        self.LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        self.resW, self.resH = args.resolution.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del (self.labels[0])

        # Load Tensorflow Lite Model
        self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # Initialize video stream
        self.videostream = VideoStream(resolution=(self.imW, self.imH), framerate=30).start()
        time.sleep(1)

