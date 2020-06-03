# Packages needed: opencv-python, opencv-contrib-python
import cv2
import numpy as np

def drawBoundingBox(img, bbox):
    # bbox is (x, y, w, h)
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)
    cv2.putText(img, "Object Tracking", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

if __name__ == "__main__":
    # Open the video file
    cap = cv2.VideoCapture("three_cups_and_a_ball.mp4")
    # Get the video writer initialized to save the output video
    outputFile = "cups_out.avi"
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    success, img = cap.read()
    # Create a CSRT tracking object
    tracker = cv2.TrackerMOSSE_create()
    # Currently, user selects the bbox size, can make this in script using: bbox = (253.79859924316406, 79.94461059570312, 218.28109741210938, 289.8338317871094)
    bbox = cv2.selectROI("Object to Track", img, False)
    tracker.init(img, bbox)

    while True:
        success, img = cap.read()
        # Stop the program if reached end of video
        if not success:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv2.waitKey(3000)
            break

        success, bbox = tracker.update(img)

        # Tracker is still tracking the object
        if success:
            drawBoundingBox(img, bbox)
        else:
            cv2.putText(img, "Object Lost", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show image
        cv2.imshow("Object Tracker", img)
        vid_writer.write(img.astype(np.uint8))

        # Quit when press Q key
        if cv2.waitKey(1) & 0xff == ord('q'):
            break