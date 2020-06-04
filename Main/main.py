import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image
        
# Load names of classes
classesFile = "./YOLO/classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./YOLO/darknet-yolov3.cfg"
modelWeights = "./YOLO/darknet-yolov3_3000.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Used to indicate when to start tracking the cup
tracking = False

# Location of the ball
ballPreviouslyFound = False
ballBbox = None

# All possible bounding boxes
possibleCupBboxes = []

def drawBoundingBox(frame, bbox):
    # bbox is (x, y, w, h)
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    global tracking, ballPreviouslyFound, ballBbox, possibleCupBboxes
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    ballFound = False
    possibleCupBboxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        objectName = classes[classIds[i]]
        # There are multiple cups, but only one ball
        if objectName == "cup":
            possibleCupBbox = (left, top, width, height)
            possibleCupBboxes.append(possibleCupBbox)
        elif objectName == "ball":
            ballFound = True
            ballPreviouslyFound = True
            ballBbox = (left, top, width, height)
        # drawBoundingBox(frame, (left, top, width, height))
        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # Ball has disappeared
    if not ballFound and ballPreviouslyFound:
        tracking = True

fileName = "three_cups_video"

if __name__ == "__main__":
    # Open the video file
    cap = cv.VideoCapture("src/" + fileName + ".mp4")
    # UNCOMMENT THIS TO SWITCH TO WEBCAM INPUT
    # cap = cv.VideoCapture(0)
    # Get the video writer initialized to save the output video
    outputFile = "src/" + fileName + "_out.avi"
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    # Run object recognition every 5 frames (to speed up processing)
    frameCount = 0

    # Read the first frame
    success, currFrame = cap.read()
    prevFrame = currFrame
    prevCount = 0

    # Create a KCF tracking object
    tracker = cv.TrackerKCF_create()
    trackingSetup = False
    cupFound = False
    bbox = None
    gameOver = False

    while True:
        # Read a new frame
        success, currFrame = cap.read()
        frameCount += 1

        # Stop the program if reached end of video
        if not success:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break
        
        # If not tracking yet, use object recognition and 
        # process every 5 frames to check if the ball has disappeared into a cup
        if not tracking and frameCount % 5 == 0:
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(currFrame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            # Sets the input to the network
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))
            # Remove the bounding boxes with low confidence
            postprocess(currFrame, outs)

        # Else if tracking, track the cup every frame
        elif tracking:
            # Setup tracking
            if not trackingSetup:
                ballx = ballBbox[0] + ballBbox[2] / 2
                for possibleCupBbox in possibleCupBboxes:
                    cupx1 = possibleCupBbox[0]
                    cupx2 = possibleCupBbox[0] + possibleCupBbox[2]
                    # If ball was last seen in a cup
                    if cupx1 < ballx and cupx2 > ballx:
                        bbox = possibleCupBbox
                        tracker.init(currFrame, bbox)
                        cupFound = True
                        break
                trackingSetup = True

            # If we found a cup during the tracking setup
            if cupFound:
                # Game is over when no motion (or no difference between frames)
                diff = cv.absdiff(prevFrame, currFrame)
                count = np.sum(diff)
                if abs(count - prevCount) < 2000:
                    gameOver = True
                # When the game is over, it's over for the rest of the video
                if gameOver:
                    drawBoundingBox(currFrame, bbox)
                    cv.putText(currFrame, "Ball Is Located Here!", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # While the game is not over, continue tracking the ball
                else:
                    prevFrame = currFrame.copy()
                    prevCount = count
                    # Track the cup
                    success, bbox = tracker.update(currFrame)
                    cv.putText(currFrame, "Tracking Cup", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    # If tracker is still tracking the object
                    if success:
                        drawBoundingBox(currFrame, bbox)
            else:
                cv.putText(currFrame, "Could Not Find Cup", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display frame
        cv.imshow("Three Cups and a Ball", currFrame)
        # Write frame to video
        vid_writer.write(currFrame.astype(np.uint8))

        # Quit when press Q key
        if cv.waitKey(1) & 0xff == ord('q'):
            break