import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-l", "--classes", required=True, help="path to coco object classes")
ap.add_argument("-s", "--config", required=True, help="path to yolo config")
ap.add_argument("-w", "--weights", required=True, help="path to yolo pre-trained model weights")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(args["config"], args["weights"])
classes = []
with open(args["classes"], "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# different random colors
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

outputLayers = net.getLayerNames()
outputLayers = [outputLayers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(outputLayers)

# print(image.shape)
# print(type(blob))
# print(np.shape(blob))

# Show blobs
# for img in blob:
#     print(np.shape(img))
#     for (n, im) in enumerate(img):
#         print(np.shape(im))
#         cv2.imshow(str(n), im)
#         # cv2.waitKey(0)
# cv2.waitKey(0)

# Show input image
# cv2.namedWindow('window', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('window', 1200, 800)
# image = cv2.resize(image, (1200,800))
# cv2.imshow('window', image)
# cv2.waitKey(0)

boxes = []
confidences = []
classIDs = []
# loop over each ouput in the output layer
for output in layerOutputs:
    # loop over each detection in an output
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            x = int(centerX - width/2)
            y = int(centerY - height/2)

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

goodindxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

if len(goodindxs) > 0:
    for i in goodindxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in colors[classIDs[i]]]

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        text = "{}-{:.4f}%".format(classes[classIDs[i]], confidences[i]*100)
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', 1200, 800)
image = cv2.resize(image, (1200,800))
cv2.imshow('window', image)
cv2.waitKey(0)


