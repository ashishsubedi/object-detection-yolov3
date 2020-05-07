import cv2
import numpy as np
from imutils.video import VideoStream


# Load Model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# Read imgs
# img = cv2.imread('cameraman.tif')

# Read Labels
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
CONFIDENCE = 0.5
NMS_THRESH = 0.3

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# Setting Output Layers
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Enabling Webcam Capture
# cap = cv2.VideoCapture(0)
cap = VideoStream(0).start()


while(True):
    # ret, img = cap.read()
    img = cap.read()
    (H, W) = img.shape[:2]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convert imgs to Blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    # Perform Detection
    net.setInput(blob)
    ouptuts = net.forward(ln)

    # Draw bbox around objects
    boxes = []
    confidences = []
    classIDs = []

    for ouptut in ouptuts:
        for detection in ouptut:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = detection[4]
            if(confidence > CONFIDENCE):
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classId)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESH)

    if len(idxs) > 0:
            # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the img
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
# show the output img
    cv2.imshow("img", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.stop()
cap.stream.release()
