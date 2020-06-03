# Packages needed: opencv-python, opencv-contrib-python
import cv2

def drawBoundingBox(img, bbox):
    # bbox is (x, y, w, h)
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    print(bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)
    cv2.putText(img, "Object Tracking", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

if __name__ == "__main__":
    # Read from webcam
    capture = cv2.VideoCapture(0)
    success, img = capture.read()
    # Create a CSRT tracking object
    tracker = cv2.TrackerCSRT_create()
    # Currently, user selects the bbox size, can make this in script using: bbox = (253.79859924316406, 79.94461059570312, 218.28109741210938, 289.8338317871094)
    bbox = cv2.selectROI("Object to Track", img, False)
    tracker.init(img, bbox)

    while True:
        success, img = capture.read()
        success, bbox = tracker.update(img)

        # Tracker is still tracking the object
        if success:
            drawBoundingBox(img, bbox)
        else:
            cv2.putText(img, "Object Lost", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show image
        cv2.imshow("Object Tracker", img)

        # Quit when press Q key
        if cv2.waitKey(1) & 0xff == ord('q'):
            break