
import os
import sys
import cv2

# * custom imports
from draw import drawRectangle, drawText
from tracker import createTrackerByName

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

print(f'\n {os.getcwd()} \n')

# load object detection dnn from pre-trained model
net = cv2.dnn.readNetFromCaffe(
    'Facial_Tracking/models/object_detection/deploy.prototxt',
    'Facial_Tracking/models/object_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# in_width and in_height are the dimensions of the input image
in_width = 300
in_height = 300
# mean is a list of 3 values representing the mean RGB values of the input image
mean = [104, 117, 123]
# conf_threshold is the confidence threshold for the model's output
conf_threshold = 0.7

# loop through frames until ESC is pressed
while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()

    if not has_frame:
        break

    # Start timer
    timer = cv2.getTickCount()

    frame= cv2.flip(frame, 1)

    # Get frame width and height
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)

    # Run the model
    net.setInput(blob)
    detections = net.forward()

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_width)

            # draw rectangle
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))

            # draw text
            label = "Confidence: %.4f" % confidence
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]), (x_left_bottom + label_size[0], y_left_bottom + baseline), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    # Display FPS on frame
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
