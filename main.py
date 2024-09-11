import os
import sys
import cv2
import numpy as np
from tarfile import TarFile
from urllib.request import urlretrieve
from Facial_Tracking.draw import drawRectangle, drawText

def createTrackerByName(tracker_type):
    # Create a tracker based on tracker name
    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    else:
        tracker = cv2.TrackerMOSSE_create()

    return tracker



# model path
modelPath = '/20180408-102900/20180408-102900.pb'

# Set up tracker types
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# capture video from camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

# create window to display frames
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# initialize object detection tracker, there are multiple types
tracker = createTrackerByName(tracker_types[3])

init = False

# load object detection dnn from pre-trained model
# net = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v1_coco_2017_11_17.pb', 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt')

# width
width = 300
# height
height = 300
# mean
mean = [0,0,0]
# confidence
conf = .7   

# loop through frames until ESC is pressed
while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()

    if not init:
        # TODO: instead of having the user draw a bounding box around their face, use facial tracking instead
        # TODO  That way if the object detection efver fails the facial detection will kick in again and the face will be tracked again.
        init = True
        bbox = cv2.selectROI(win_name, frame)
        tracker.init(frame, bbox)

    if not has_frame:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))
        # TODO: if failing, use facial detection to start backup again

    # Display Info
    drawText(frame, str(tracker) + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))


    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
